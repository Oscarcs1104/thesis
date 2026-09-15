"""MoLA cross-layer fusion over PRETRAINED backbones, for the property-prediction ablation.

Same fusion mechanism as HybridMoLA -- one pooled token per layer per modality, stacked,
cross-attended, then a learned weighted sum -- but both branches start from published
pretrained weights instead of from scratch:

    graph branch     Hu et al. 2020 GIN          (crossmodal_model/model/pretrained_gin.py)
    language branch  ChemBERTa or MoLFormer      (crossmodal_model/model/pretrained_lm.py)

use_graph / use_lm give the four configurations of the ablation:

    lm only         graph off, ChemBERTa or MoLFormer
    lm + GIN        both on

tokens_per_layer drops from 2 to 1 when a branch is off, so layer_weights is sized to
what is actually present -- a disabled branch contributes no parameters at all, rather
than contributing zeros that would still be counted and still be trained.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from crossmodal_model.model.pretrained_gin import PretrainedGIN
from crossmodal_model.model.pretrained_lm import PretrainedLanguageEncoder


class PretrainedMoLA(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        output_dim: int = 1,
        num_layers: int = 3,
        use_graph: bool = True,
        use_lm: bool = True,
        lm_key: str = "chemberta",
        lm_freeze: bool = True,
        lm_max_length: int = 128,
        gin_variant: str = "contextpred",
        gin_freeze: bool = False,
        gin_num_layer: int = 5,
        gin_pool: str = "mean",
        gin_dropout: float = 0.5,
        head_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if not use_graph and not use_lm:
            raise ValueError("PretrainedMoLA needs at least one of use_graph / use_lm")
        self.use_graph, self.use_lm = use_graph, use_lm
        self.num_layers = num_layers

        if use_graph:
            # The GIN has 5 pretrained layers; MoLA fuses num_layers slots, so the last
            # num_layers are taken. Truncating the stack instead would throw away
            # pretrained weights.
            self.gin = PretrainedGIN(
                hidden_dim=hidden_dim, num_layer=gin_num_layer, drop_ratio=gin_dropout,
                pool=gin_pool, variant=gin_variant, freeze=gin_freeze,
            )
        if use_lm:
            self.lm = PretrainedLanguageEncoder(
                hidden_dim=hidden_dim, model_key=lm_key, num_layers=num_layers,
                freeze=lm_freeze, max_length=lm_max_length,
            )

        self.tokens_per_layer = int(use_graph) + int(use_lm)
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8)
        self.layer_weights = nn.Parameter(torch.ones(num_layers * self.tokens_per_layer, 1, 1))
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(4 * hidden_dim, output_dim),
        )

    def _graph_tokens(self, data) -> List[torch.Tensor]:
        _, layer_states = self.gin(data.x, data.edge_index, data.edge_attr, data.batch)
        return layer_states[-self.num_layers:]

    def forward(self, data):
        tokens_per_layer: List[List[torch.Tensor]] = []
        graph_tokens = self._graph_tokens(data) if self.use_graph else None
        lm_tokens = None
        if self.use_lm:
            device = data.batch.device if hasattr(data, "batch") else data.x.device
            lm_tokens = self.lm(data.smiles, device)

        for i in range(self.num_layers):
            slot = []
            if graph_tokens is not None:
                slot.append(graph_tokens[i])
            if lm_tokens is not None:
                slot.append(lm_tokens[i])
            tokens_per_layer.append(torch.stack(slot, dim=0))

        fused_all = torch.cat(tokens_per_layer, dim=0)           # [L*T, B, H]
        attn_out, attn_map = self.cross_attention(fused_all, fused_all, fused_all)
        fused_feat = (attn_out * self.layer_weights).sum(dim=0)  # [B, H]
        return [fused_feat, self.layer_weights.view(-1, 1, 1), attn_map, self.head(fused_feat)]


def build_config(name: str, **kwargs) -> PretrainedMoLA:
    """The four rows of the ablation table, by name."""
    configs = {
        "chemberta":       dict(use_graph=False, use_lm=True, lm_key="chemberta"),
        "molformer":       dict(use_graph=False, use_lm=True, lm_key="molformer"),
        "chemberta+gin":   dict(use_graph=True, use_lm=True, lm_key="chemberta"),
        "molformer+gin":   dict(use_graph=True, use_lm=True, lm_key="molformer"),
        "gin":             dict(use_graph=True, use_lm=False),
    }
    if name not in configs:
        raise ValueError(f"config must be one of {sorted(configs)}, got {name!r}")
    return PretrainedMoLA(**{**configs[name], **kwargs})


CONFIGS = ("chemberta", "molformer", "chemberta+gin", "molformer+gin", "gin")

__all__ = ["PretrainedMoLA", "build_config", "CONFIGS"]
