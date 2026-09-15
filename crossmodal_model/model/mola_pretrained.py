"""MoLA cross-layer fusion over PRETRAINED backbones, for the property-prediction ablation.

Same fusion as HybridMoLA -- one pooled token per layer per modality, stacked,
cross-attended, weighted-summed -- but both branches start from published pretrained
weights instead of from scratch:

    graph branch     Hu et al. 2020 GIN        (crossmodal_model/model/pretrained_gin.py)
    language branch  ChemBERTa or MoLFormer    (crossmodal_model/model/pretrained_lm.py)

use_graph / use_lm give the rows of the ablation. tokens_per_layer drops from 2 to 1 when
a branch is off, so layer_weights is sized to what is present: a disabled branch
contributes no parameters at all, rather than zeros that are still counted and trained.

The forward is split into encode() and fuse() on purpose. When both backbones are frozen
their per-layer states are a deterministic function of the molecule, so encode() runs
ONCE per dataset (precompute_features) and every epoch replays fuse() over cached
tensors. On a 100-epoch sweep that is the difference between re-running a 77M transformer
a hundred times over the same molecules and running it once. The projections live here,
not in the backbones, which is what makes the cached tensors backbone-native and the
trainable part small.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from crossmodal_model.model.pretrained_gin import PretrainedGIN
from crossmodal_model.model.pretrained_lm import PretrainedLanguageEncoder

CONFIGS = ("chemberta", "molformer", "gin", "chemberta+gin", "molformer+gin")

_CONFIG_SPECS = {
    "chemberta":     dict(use_graph=False, use_lm=True, lm_key="chemberta"),
    "molformer":     dict(use_graph=False, use_lm=True, lm_key="molformer"),
    "gin":           dict(use_graph=True, use_lm=False),
    "chemberta+gin": dict(use_graph=True, use_lm=True, lm_key="chemberta"),
    "molformer+gin": dict(use_graph=True, use_lm=True, lm_key="molformer"),
}


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
        gin_freeze: bool = True,
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
            self.gin = PretrainedGIN(
                num_layer=gin_num_layer, drop_ratio=gin_dropout, pool=gin_pool,
                variant=gin_variant, freeze=gin_freeze,
            )
            self.graph_proj = nn.Linear(self.gin.emb_dim, hidden_dim)
        if use_lm:
            self.lm = PretrainedLanguageEncoder(
                model_key=lm_key, num_layers=num_layers, freeze=lm_freeze,
                max_length=lm_max_length,
            )
            self.lm_proj = nn.Linear(self.lm.lm_hidden, hidden_dim)

        self.tokens_per_layer = int(use_graph) + int(use_lm)
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8)
        self.layer_weights = nn.Parameter(torch.ones(num_layers * self.tokens_per_layer, 1, 1))
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(4 * hidden_dim, output_dim),
        )

    @property
    def backbones_frozen(self) -> bool:
        ok = True
        if self.use_graph:
            ok = ok and self.gin.frozen
        if self.use_lm:
            ok = ok and self.lm.freeze
        return ok

    def encode(self, data) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Backbone states only, at their own widths. [B, num_layers, emb] per branch."""
        graph = lm = None
        if self.use_graph:
            _, states = self.gin(data.x, data.edge_index, data.edge_attr, data.batch)
            # The GIN has 5 pretrained layers and MoLA fuses num_layers slots; take the
            # last ones rather than truncating the stack, which would discard weights.
            graph = torch.stack(states[-self.num_layers:], dim=1)
        if self.use_lm:
            device = data.x.device
            lm = torch.stack(self.lm(data.smiles, device), dim=1)
        return graph, lm

    def fuse(self, graph: Optional[torch.Tensor], lm: Optional[torch.Tensor]):
        slots: List[torch.Tensor] = []
        for i in range(self.num_layers):
            layer = []
            if graph is not None:
                layer.append(self.graph_proj(graph[:, i]))
            if lm is not None:
                layer.append(self.lm_proj(lm[:, i]))
            slots.append(torch.stack(layer, dim=0))
        fused_all = torch.cat(slots, dim=0)                      # [L*T, B, H]
        attn_out, attn_map = self.cross_attention(fused_all, fused_all, fused_all)
        fused_feat = (attn_out * self.layer_weights).sum(dim=0)  # [B, H]
        return [fused_feat, self.layer_weights.view(-1, 1, 1), attn_map, self.head(fused_feat)]

    def forward(self, data):
        return self.fuse(*self.encode(data))


@torch.no_grad()
def precompute_features(model: PretrainedMoLA, loader, device) -> Dict[str, torch.Tensor]:
    """Run the frozen backbones once over a loader and keep their per-layer states.

    Only valid when every backbone is frozen, which is checked: with a trainable backbone
    the states change every step and a cache would silently train on stale features. The
    loader must not shuffle, so the cached rows line up with the targets.
    """
    if not model.backbones_frozen:
        raise ValueError("precompute_features needs every backbone frozen; a trainable "
                         "backbone's states change each step and the cache would go stale")
    model.eval()
    graph_parts, lm_parts, targets = [], [], []
    for batch in loader:
        batch = batch.to(device)
        graph, lm = model.encode(batch)
        if graph is not None:
            graph_parts.append(graph.cpu())
        if lm is not None:
            lm_parts.append(lm.cpu())
        targets.append(batch.y.float().view(-1).cpu())
    out: Dict[str, torch.Tensor] = {"y": torch.cat(targets)}
    if graph_parts:
        out["graph"] = torch.cat(graph_parts)
    if lm_parts:
        out["lm"] = torch.cat(lm_parts)
    return out


def build_config(name: str, **kwargs) -> PretrainedMoLA:
    """The rows of the ablation table, by name."""
    if name not in _CONFIG_SPECS:
        raise ValueError(f"config must be one of {sorted(_CONFIG_SPECS)}, got {name!r}")
    spec = dict(_CONFIG_SPECS[name])
    # A config with no language branch must not be handed an lm_key, and the reverse.
    if not spec.get("use_lm", False):
        kwargs.pop("lm_key", None)
    return PretrainedMoLA(**{**spec, **kwargs})


__all__ = ["PretrainedMoLA", "build_config", "precompute_features", "CONFIGS"]
