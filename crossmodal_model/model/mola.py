import torch
import torch.nn as nn

from crossmodal_model.model.encoder import Encoder


class MoLA(nn.Module):
    """
    MoLA backbone model: first extract the modality fusion features of each layer through
    Encoder, then perform cross-layer multi-head attention, weighted fusion and output the
    final prediction.

    Thesis audit note: Graph + SMILES only -- the original MoLA fp+MolFormer (DNM) branch
    has been removed entirely.
    """
    def __init__(self, graph_dim, sm_vocab_size, hidden_dim, output_dim, num_layers, positional_smiles=False, max_sm_len=100):
        super(MoLA, self).__init__()
        self.encoder = Encoder(graph_dim, sm_vocab_size, hidden_dim, num_layers, positional_smiles=positional_smiles, max_sm_len=max_sm_len)
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8)

        # Output layer after fusion
        self.out_layer_final = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4 * hidden_dim, output_dim),
        )
        # Weighting parameters for each channel (layer x 2 tokens per layer: graph, sm)
        self.layer_weights = nn.Parameter(torch.ones(num_layers * 2, 1, 1))

    def _data_fields(self, data):
        return data.x, data.edge_index, data.sm, data.batch

    def encode_for_generation(self, data):
        """crossmodal_model/generation/decoder.py's entry point: the shared encoder's raw
        (unpooled) graph-node + SMILES-char states, for a cross-attention decoder to
        condition on -- see Encoder.forward_with_raw. Requires the model to have been built
        with positional_smiles=True."""
        x, edge_index, sm, batch = self._data_fields(data)
        _, raw = self.encoder.forward_with_raw(x, edge_index, sm, batch)
        return raw

    def forward(self, data):
        x, edge_index, sm, batch = self._data_fields(data)

        # Get the outputs of all layers [T, B, H]，T = num_layers * 2
        fused_all = self.encoder(x, edge_index, sm, batch)

        # Cross-layer multi-head attention
        attn_out, attn_map = self.cross_attention(fused_all, fused_all, fused_all)
        # Weighted sum fusion
        fused_feat = (attn_out * self.layer_weights).sum(dim=0)  # [B, H]
        # Final Output
        fused_out_final = self.out_layer_final(fused_feat)       # [B, output_dim]

        # [ fused_feature, layer_weights.view(-1,1,1), attn_map, fused_out_final ]
        return [
            fused_feat,
            self.layer_weights.view(-1, 1, 1),
            attn_map,
            fused_out_final
        ]
