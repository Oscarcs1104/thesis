"""Ablation replicating how MoLA (C:\\CVAIL\\Thesis\\MoLA) used IBM's MoLFormer:
a FROZEN, PRECOMPUTED embedding (data_pipeline/precompute_molformer_embeddings.py)
concatenated with the graph representation -- no live HuggingFace forward pass
happens during training at all (unlike model/encoders.py's LanguageEncoder,
which re-runs the text backbone every batch). Predictor-only, no decoder.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

try:
    from .encoders import GraphEncoder
except Exception:
    from encoders import GraphEncoder


class GraphPrecomputedMolformerModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        molformer_dim: int = 768,
        graph_backbone: str = "gin",
        num_layers: int = 3,
        dropout: float = 0.3,
        node_encoding: str = "dense",
        node_vocab_sizes: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.graph_encoder = GraphEncoder(
            hidden_dim=hidden_dim,
            graph_backbone=graph_backbone,
            num_layers=num_layers,
            dropout=dropout,
            node_encoding=node_encoding,
            node_vocab_sizes=node_vocab_sizes,
        )
        self.molformer_proj = nn.Linear(molformer_dim, hidden_dim)

        fused_dim = hidden_dim * 2
        self.head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, output_dim),
        )

    def forward(self, data: torch.nn.Module) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        molformer_emb = data.molformer_emb  # (B, molformer_dim) -- precomputed & frozen, never run here
        edge_attr = getattr(data, "edge_attr", None)

        _, layer_graph_states = self.graph_encoder(x, edge_index, batch, edge_attr=edge_attr)
        graph_state = layer_graph_states[-1]
        molformer_state = self.molformer_proj(molformer_emb.float())

        fused = torch.cat([graph_state, molformer_state], dim=-1)
        return self.head(fused)


def build_precomputed_molformer_model_from_args(args) -> GraphPrecomputedMolformerModel:
    return GraphPrecomputedMolformerModel(
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        molformer_dim=args.molformer_dim,
        graph_backbone=args.graph_backbone,
        num_layers=args.num_layers,
        dropout=args.dropout,
        node_encoding=args.node_encoding,
        node_vocab_sizes=args.node_vocab_sizes,
    )
