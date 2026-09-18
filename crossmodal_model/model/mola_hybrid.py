import torch
import torch.nn as nn

from crossmodal_model.model.encoder_hybrid import HybridEncoder


class HybridMoLA(nn.Module):
    """MoLA's cross-layer attention fusion + regression head + generation hookup
    (encode_for_generation), built on HybridEncoder instead of the original from-scratch
    graph branch. use_graph/use_smiles: single-modality ablation, see encoder_hybrid.py."""

    def __init__(
        self,
        sm_vocab_size,
        hidden_dim,
        output_dim,
        num_layers,
        positional_smiles: bool = False,
        max_sm_len: int = 100,
        graph_backbone: str = "gin",
        node_encoding: str = "categorical",
        node_vocab_sizes=None,
        edge_vocab_sizes=None,
        graph_pooling: str = "add",
        graph_dropout: float = 0.3,
        use_graph: bool = True,
        use_smiles: bool = True,
        gin_hidden_mult: int = 1,
    ) -> None:
        super().__init__()
        self.encoder = HybridEncoder(
            sm_vocab_size=sm_vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            positional_smiles=positional_smiles,
            max_sm_len=max_sm_len,
            graph_backbone=graph_backbone,
            node_encoding=node_encoding,
            node_vocab_sizes=node_vocab_sizes,
            edge_vocab_sizes=edge_vocab_sizes,
            graph_pooling=graph_pooling,
            graph_dropout=graph_dropout,
            use_graph=use_graph,
            use_smiles=use_smiles,
            gin_hidden_mult=gin_hidden_mult,
        )
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8)
        self.out_layer_final = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4 * hidden_dim, output_dim),
        )
        self.layer_weights = nn.Parameter(torch.ones(num_layers * self.encoder.tokens_per_layer, 1, 1))

    def _data_fields(self, data):
        return data.x, data.edge_index, data.edge_attr, data.sm, data.batch

    def encode_for_generation(self, data, with_fusion: bool = False):
        """Raw per-atom and per-character states for a cross-attention decoder.

        with_fusion also returns the cross-layer fusion's output, [B, 2L, H], so the
        decoder can attend to it alongside the raw states.

        It is off by default because that is how every result so far was measured, and
        because it was not an oversight that the fusion sat outside this path: a decoder
        reconstructing a molecule needs to know which atom is where, and 2L pooled
        vectors cannot say that. The raw states are the right memory.

        What the default does mean is that cross_attention and layer_weights -- the MoLA
        mechanism this architecture is named for -- never run during generation, so an
        ablation over this path compares which raw states enter the memory and says
        nothing about the fusion. Turning it on puts the fusion in the graph and makes
        that comparison possible.
        """
        x, edge_index, edge_attr, sm, batch = self._data_fields(data)
        fused_all, raw = self.encoder.forward_with_raw(x, edge_index, edge_attr, sm, batch)
        if not with_fusion:
            return raw
        # attn_out * layer_weights, not the summed vector: the sum is one token among
        # roughly 130 and trivial for the decoder to ignore -- the same failure the
        # original prepended property token had. Keeping the 2L tokens separate gives the
        # fusion real presence in the memory, and routes the gradient through both
        # cross_attention and layer_weights rather than only the first.
        attn_out, _ = self.cross_attention(fused_all, fused_all, fused_all)
        weighted = attn_out * self.layer_weights          # [2L, B, H]
        raw = dict(raw)
        raw["fused_state"] = weighted.transpose(0, 1)     # [B, 2L, H]
        return raw

    def forward(self, data):
        x, edge_index, edge_attr, sm, batch = self._data_fields(data)
        fused_all = self.encoder(x, edge_index, edge_attr, sm, batch)
        attn_out, attn_map = self.cross_attention(fused_all, fused_all, fused_all)
        fused_feat = (attn_out * self.layer_weights).sum(dim=0)
        fused_out_final = self.out_layer_final(fused_feat)
        return [fused_feat, self.layer_weights.view(-1, 1, 1), attn_map, fused_out_final]
