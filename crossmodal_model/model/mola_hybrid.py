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

    def encode_for_generation(self, data):
        x, edge_index, edge_attr, sm, batch = self._data_fields(data)
        _, raw = self.encoder.forward_with_raw(x, edge_index, edge_attr, sm, batch)
        return raw

    def forward(self, data):
        x, edge_index, edge_attr, sm, batch = self._data_fields(data)
        fused_all = self.encoder(x, edge_index, edge_attr, sm, batch)
        attn_out, attn_map = self.cross_attention(fused_all, fused_all, fused_all)
        fused_feat = (attn_out * self.layer_weights).sum(dim=0)
        fused_out_final = self.out_layer_final(fused_feat)
        return [fused_feat, self.layer_weights.view(-1, 1, 1), attn_map, fused_out_final]
