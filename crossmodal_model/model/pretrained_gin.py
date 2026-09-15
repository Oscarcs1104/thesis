"""Hu et al. 2020 pretrained GIN, reimplemented for modern PyTorch Geometric.

The upstream implementation (snap-stanford/pretrain-gnns, chem/model.py) targets the 2019
PyG API: propagate() took aggr positionally and add_self_loops returned a bare tensor.
Neither survives PyG 2.x, so the layer is rewritten here. Every module name is preserved
exactly -- x_embedding1/2, gnns.{i}.mlp.{0,2}, gnns.{i}.edge_embedding1/2, batch_norms.{i}
-- so the released state_dict loads unchanged.

Verified against contextpred.pth: 57 tensors, 5 layers, emb_dim 300, 1,860,905 parameters.

Input must use the Hu et al. schema (data_pipeline/features_pretrain_gnn.py), NOT our OGB
9+3 one: these embeddings were trained on those exact indices, and feeding different ones
would quietly mean nothing.

Checkpoints (7.5 MB each) live in the repo itself, under
chem/model_gin/<variant>.pth, with <variant> in contextpred, edgepred, infomax, masking,
supervised, supervised_contextpred, supervised_edgepred, supervised_infomax,
supervised_masking.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple
from urllib.request import urlretrieve

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import add_self_loops

from data_pipeline.features_pretrain_gnn import (
    NUM_ATOM_TYPE,
    NUM_BOND_DIRECTION,
    NUM_BOND_TYPE,
    NUM_CHIRALITY_TAG,
)

_BASE_URL = "https://raw.githubusercontent.com/snap-stanford/pretrain-gnns/master/chem/model_gin"
PRETRAINED_VARIANTS = (
    "contextpred", "edgepred", "infomax", "masking", "supervised",
    "supervised_contextpred", "supervised_edgepred", "supervised_infomax", "supervised_masking",
)
_POOLS = {"add": global_add_pool, "mean": global_mean_pool, "max": global_max_pool}


def download_pretrained_gin(variant: str = "contextpred",
                            cache_dir: str = "checkpoints/pretrained_gin") -> Path:
    if variant not in PRETRAINED_VARIANTS:
        raise ValueError(f"variant must be one of {PRETRAINED_VARIANTS}, got {variant!r}")
    out = Path(cache_dir) / f"{variant}.pth"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading pretrained GIN '{variant}' -> {out}")
        urlretrieve(f"{_BASE_URL}/{variant}.pth", out)
    return out


class GINConv(MessagePassing):
    """GIN with edge features folded into the message, as Hu et al. define it.

    Self-loops carry bond type 4 and direction 0, matching the upstream convention: the
    pretrained edge embedding has a trained row for that token, so omitting the self
    loops would change what every other row means.
    """

    def __init__(self, emb_dim: int) -> None:
        super().__init__(aggr="add")
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim), nn.ReLU(), nn.Linear(2 * emb_dim, emb_dim)
        )
        self.edge_embedding1 = nn.Embedding(NUM_BOND_TYPE, emb_dim)
        self.edge_embedding2 = nn.Embedding(NUM_BOND_DIRECTION, emb_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = torch.zeros(x.size(0), 2, dtype=edge_attr.dtype, device=edge_attr.device)
        self_loop_attr[:, 0] = 4
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
        edge_emb = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        return self.propagate(edge_index, x=x, edge_attr=edge_emb)

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return x_j + edge_attr

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        return self.mlp(aggr_out)


class PretrainedGIN(nn.Module):
    """The pretrained GIN, exposing every layer's pooled state for MoLA-style fusion.

    `proj` maps the frozen 300-dim space onto the fusion width, so the backbone keeps its
    own dimensionality and only the projection is learned when the backbone is frozen.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layer: int = 5,
        emb_dim: int = 300,
        drop_ratio: float = 0.5,
        pool: str = "mean",
        variant: Optional[str] = "contextpred",
        checkpoint_path: Optional[str] = None,
        freeze: bool = False,
        cache_dir: str = "checkpoints/pretrained_gin",
    ) -> None:
        super().__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.drop_ratio = drop_ratio
        if pool not in _POOLS:
            raise ValueError(f"pool must be one of {sorted(_POOLS)}")
        self.pool = pool

        self.x_embedding1 = nn.Embedding(NUM_ATOM_TYPE, emb_dim)
        self.x_embedding2 = nn.Embedding(NUM_CHIRALITY_TAG, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        self.gnns = nn.ModuleList([GINConv(emb_dim) for _ in range(num_layer)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(num_layer)])

        self.loaded_from: Optional[str] = None
        if checkpoint_path or variant:
            path = Path(checkpoint_path) if checkpoint_path else download_pretrained_gin(variant, cache_dir)
            self.load_pretrained(path)

        self.frozen = freeze
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

        # Learned regardless of freezing: the pretrained space is 300-dim, the fusion
        # runs at hidden_dim.
        self.proj = nn.Linear(emb_dim, hidden_dim)

    def load_pretrained(self, path: Path) -> None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        missing, unexpected = self.load_state_dict(state, strict=False)
        # Loud on purpose. A silent key mismatch means randomly initialized weights
        # posing as a pretrained backbone, and then every number in the ablation table
        # is about a model nobody trained.
        real_missing = [k for k in missing if not k.startswith("proj.")]
        if real_missing or unexpected:
            raise RuntimeError(
                f"pretrained GIN did not load cleanly from {path}\n"
                f"  missing:    {real_missing}\n"
                f"  unexpected: {list(unexpected)}"
            )
        self.loaded_from = str(path)

    def train(self, mode: bool = True):
        """A frozen backbone must never re-enter train(): its BatchNorm running stats
        would drift away from what it was pretrained with."""
        super().train(mode)
        if getattr(self, "frozen", False):
            for m in (self.x_embedding1, self.x_embedding2, self.gnns, self.batch_norms):
                m.eval()
        return self

    def forward(self, x, edge_index, edge_attr, batch) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Returns (final node states projected to hidden_dim, per-layer pooled states)."""
        if x.dtype != torch.long or x.size(1) != 2:
            raise ValueError(
                f"PretrainedGIN expects the Hu et al. schema: long x of shape [N, 2], got "
                f"{x.dtype} {tuple(x.shape)}. Featurize with "
                f"data_pipeline.features_pretrain_gnn.smiles_to_data_pretrain."
            )
        h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        layer_states: List[torch.Tensor] = []
        for i, (conv, bn) in enumerate(zip(self.gnns, self.batch_norms)):
            h = bn(conv(h, edge_index, edge_attr))
            # Upstream drops the ReLU on the last layer; keeping it would change the
            # representation the checkpoint was trained to produce.
            if i == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            layer_states.append(self.proj(_POOLS[self.pool](h, batch)))
        return self.proj(h), layer_states


__all__ = ["PretrainedGIN", "GINConv", "download_pretrained_gin", "PRETRAINED_VARIANTS"]
