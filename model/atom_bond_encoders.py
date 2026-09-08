"""Learned embeddings for the OGB-style integer atom/bond feature indices.

`AtomEncoder(x)`  : x  [N, 9] long  -> [N, hidden_dim]   (sum of 9 column embeddings)
`BondEncoder(ea)` : ea [E, 3] long  -> [E, hidden_dim]   (sum of 3 column embeddings)

Same design as `ogb.graphproppred.mol_encoder`, kept local so `ogb` is not a
dependency. Feature column sizes come from `data_pipeline/features.py` so the two
sides can never drift apart.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from data_pipeline.features import ATOM_FEATURE_DIMS, BOND_FEATURE_DIMS


class _ColumnEmbedding(nn.Module):
    def __init__(self, column_sizes, hidden_dim: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(int(size), hidden_dim) for size in column_sizes])
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight.data)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        feats = feats.long()
        if feats.dim() == 1:
            feats = feats.unsqueeze(-1)
        out = 0
        for i, emb in enumerate(self.embeddings):
            # non-in-place clamp: never mutate the caller's x / edge_attr, just
            # guard against an out-of-range index from a malformed cache
            out = out + emb(feats[:, i].clamp(0, emb.num_embeddings - 1))
        return out


class AtomEncoder(_ColumnEmbedding):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__(ATOM_FEATURE_DIMS, hidden_dim)


class BondEncoder(_ColumnEmbedding):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__(BOND_FEATURE_DIMS, hidden_dim)
