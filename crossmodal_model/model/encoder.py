import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool


class Encoder(nn.Module):
    """
    Multimodal Encoder (graph + SMILES), containing a FirstLayer and a (num_layers-1)
    identical MultimodalLayer.

    Thesis audit note: the original MoLA third modality (a classic Morgan fingerprint
    concatenated with a frozen MolFormer embedding, fused via a Dendritic Neuron Model)
    has been removed entirely -- not used by this thesis. The fused sequence is always
    [num_layers*2, B, H] (graph, sm per layer): Graph + SMILES only.

    forward returns a fused output sequence of shape [num_layers*2, batch_size, hidden_dim].
    """
    def __init__(self, graph_dim, sm_vocab_size, hidden_dim, num_layers, positional_smiles=False, max_sm_len=100):
        super(Encoder, self).__init__()
        self.num_layers = num_layers

        # A2 (thesis audit, SMILES-branch fix): positional_smiles=False (default) reproduces
        # the original MoLA sm_transformer bit-for-bit -- including its pre-existing
        # batch/sequence axis bug (sm_embed is [B,L,H] fed into a batch_first=False
        # TransformerEncoderLayer, so attention runs across the wrong axis: different
        # molecules at a fixed character position, not characters within one molecule).
        # positional_smiles=True fixes that (batch_first=True) plus adds a learned
        # positional embedding and a real padding mask -- the padding mask can't be
        # expressed correctly without also fixing batch_first, since src_key_padding_mask
        # is only meaningful once the layer knows which axis is actually the batch.
        self.positional_smiles = positional_smiles
        self.max_sm_len = max_sm_len
        if positional_smiles:
            self.position_embedding = nn.Embedding(max_sm_len, hidden_dim)

        def _sm_transformer():
            return nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=positional_smiles),
                num_layers=1
            )

        # First layer: receives the original input dimensions
        self.first_layer = nn.ModuleDict({
            'graph_conv': GINConv(
                nn.Sequential(
                    nn.Linear(graph_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            ),
            'sm_embedding': nn.Embedding(sm_vocab_size, hidden_dim, padding_idx=0),
            'sm_transformer': _sm_transformer(),
            'dropout': nn.Dropout(0.3),
        })

        # Subsequent layers: dimensions are all hidden_dim
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(nn.ModuleDict({
                'graph_conv': GINConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim)
                    )
                ),
                'sm_transformer': _sm_transformer(),
                'dropout': nn.Dropout(0.3),
            }))

    def _pool_sm(self, sm_x, keep):
        # keep=None (positional_smiles=False): original unmasked mean over all L positions,
        # padding included -- unchanged behavior. keep is a [B,L,1] float mask (1=real
        # token, 0=padding) otherwise: masked mean, ignoring padded positions.
        if keep is None:
            return sm_x.mean(dim=1)
        return (sm_x * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1e-6)

    def forward(self, x, edge_index, sm, batch):
        fused_all, _ = self._forward_impl(x, edge_index, sm, batch, need_raw=False)
        return fused_all

    def forward_with_raw(self, x, edge_index, sm, batch):
        """Generation entry point (crossmodal_model/generation/decoder.py): same computation
        as forward(), but also returns the LAST layer's unpooled per-node graph states and
        unpooled per-character SMILES states -- the memory a cross-attention decoder
        conditions on, instead of the pooled/fused vector the regression head uses. Requires
        positional_smiles=True (generation needs real token order and a real padding mask;
        see the note on positional_smiles above)."""
        if not self.positional_smiles:
            raise ValueError("forward_with_raw needs positional_smiles=True (generation requires a real padding mask and token order)")
        fused_all, raw = self._forward_impl(x, edge_index, sm, batch, need_raw=True)
        return fused_all, raw

    def _forward_impl(self, x, edge_index, sm, batch, need_raw=False):
        fused_out_list = []

        l0 = self.first_layer
        graph_x = F.relu(l0['graph_conv'](x, edge_index))
        graph_feat = global_mean_pool(graph_x, batch)

        pad_mask = None   # [B, L] bool, True = padding position (src_key_padding_mask)
        keep = None        # [B, L, 1] float, 1 = real token, 0 = padding (for masked pooling)
        sm_embed = l0['sm_embedding'](sm)              # [B, L, H]
        if self.positional_smiles:
            seq_len = sm.size(1)
            positions = torch.arange(seq_len, device=sm.device).unsqueeze(0).expand(sm.size(0), -1)
            sm_embed = sm_embed + self.position_embedding(positions)
            pad_mask = sm == 0
            keep = (~pad_mask).unsqueeze(-1).to(sm_embed.dtype)

        sm_x = l0['sm_transformer'](sm_embed, src_key_padding_mask=pad_mask)  # [B, L, H]
        sm_x = F.relu(sm_x)
        sm_feat = self._pool_sm(sm_x, keep)            # [B, H]

        # dropout
        graph_x = l0['dropout'](graph_x)
        sm_x = l0['dropout'](sm_x)

        fused_out_list.append(torch.stack([graph_feat, sm_feat], dim=0))

        h_x, h_sm = graph_x, sm_x
        for layer in self.layers:
            gx = F.relu(layer['graph_conv'](h_x, edge_index))
            gf = global_mean_pool(gx, batch)

            sx = layer['sm_transformer'](h_sm, src_key_padding_mask=pad_mask)
            sx = F.relu(sx)
            sf = self._pool_sm(sx, keep)

            # dropout
            gx = layer['dropout'](gx)
            sx = layer['dropout'](sx)

            fused_out_list.append(torch.stack([gf, sf], dim=0))

            h_x, h_sm = gx, sx

        # Concatenate the outputs of all layers in dimension 0
        # Result shape: [num_layers*2, batch_size, hidden_dim]
        fused_all = torch.cat(fused_out_list, dim=0)

        raw = None
        if need_raw:
            # Last layer's UNPOOLED states -- what a generation decoder cross-attends over,
            # instead of the pooled graph_feat/sm_feat that went into fused_out_list above.
            raw = {"node_state": h_x, "node_batch": batch, "sm_state": h_sm, "sm_pad_mask": pad_mask}
        return fused_all, raw
