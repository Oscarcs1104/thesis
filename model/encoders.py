from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, GINEConv, global_mean_pool

from data_pipeline.features import BOND_FEATURE_DIMS
from model.atom_bond_encoders import AtomEncoder, BondEncoder

VALID_GRAPH_BACKBONES = {"gcn", "gat", "gatv2", "gin"}
VALID_LANGUAGE_BACKBONES = {"huggingface", "none"}

# Which backbones consume bond features. GCN has no edge-feature formulation here,
# so it runs on connectivity only (documented limitation, still a valid ablation).
_EDGE_AWARE_BACKBONES = {"gin", "gat", "gatv2"}


def _load_text_tokenizer(model_name: str, trust_remote_code: bool):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    except OSError:
        # Some trust_remote_code repos declare a custom tokenizer class (via auto_map)
        # whose source file doesn't actually exist upstream (seen with
        # DeepChem/MoLFormer-c3-1.1B). Fall back to the repo's own tokenizer.json.
        import json

        from huggingface_hub import hf_hub_download
        from transformers import PreTrainedTokenizerFast

        tokenizer_file = hf_hub_download(model_name, "tokenizer.json")
        special_tokens = {}
        try:
            special_file = hf_hub_download(model_name, "special_tokens_map.json")
            with open(special_file, "r", encoding="utf-8") as handle:
                raw_special = json.load(handle)
            special_tokens = {k: (v["content"] if isinstance(v, dict) else v) for k, v in raw_special.items()}
        except Exception:
            pass
        return PreTrainedTokenizerFast(tokenizer_file=tokenizer_file, **special_tokens)


def _make_graph_conv(backbone: str, hidden_dim: int) -> Tuple[nn.Module, bool]:
    """Returns (conv, consumes_edge_features). All convs are hidden->hidden."""
    backbone = backbone.lower()
    if backbone == "gcn":
        return GCNConv(hidden_dim, hidden_dim), False
    if backbone == "gat":
        return GATConv(hidden_dim, hidden_dim, heads=4, concat=False, edge_dim=hidden_dim), True
    if backbone == "gatv2":
        return GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False, edge_dim=hidden_dim), True
    if backbone == "gin":
        mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        return GINEConv(mlp, train_eps=True), True  # edge_dim=None -> expects edge feats already at hidden_dim
    raise ValueError(f"Unsupported graph backbone: {backbone}")


class GraphEncoder(nn.Module):
    """Atom/bond-embedding + message passing. Consumes OGB-style integer features
    (``x`` [N, 9] long, ``edge_attr`` [E, 3] long). Returns the final node states
    and the mean-pooled graph vector after every layer.
    """

    def __init__(
        self,
        hidden_dim: int,
        graph_backbone: str = "gin",
        num_layers: int = 3,
        dropout: float = 0.3,
        **_ignored: Any,  # absorbs removed node_encoding / node_vocab_sizes kwargs
    ) -> None:
        super().__init__()
        graph_backbone = graph_backbone.lower()
        if graph_backbone not in VALID_GRAPH_BACKBONES:
            raise ValueError(f"graph_backbone must be one of {sorted(VALID_GRAPH_BACKBONES)}")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.graph_backbone = graph_backbone
        self.edge_aware = graph_backbone in _EDGE_AWARE_BACKBONES
        self.dropout = nn.Dropout(dropout)

        self.atom_encoder = AtomEncoder(hidden_dim)
        self.graph_layers = nn.ModuleList()
        self.bond_encoders = nn.ModuleList()
        for _ in range(num_layers):
            conv, uses_edges = _make_graph_conv(graph_backbone, hidden_dim)
            self.graph_layers.append(conv)
            self.bond_encoders.append(BondEncoder(hidden_dim) if uses_edges else nn.Identity())

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        node_state = self.atom_encoder(x)
        if edge_attr is None:
            edge_attr = torch.zeros((edge_index.size(1), len(BOND_FEATURE_DIMS)), dtype=torch.long, device=x.device)

        layer_graph_states: List[torch.Tensor] = []
        for conv, bond_encoder in zip(self.graph_layers, self.bond_encoders):
            if self.edge_aware:
                edge_emb = bond_encoder(edge_attr)
                node_state = conv(node_state, edge_index, edge_emb)
            else:
                node_state = conv(node_state, edge_index)
            node_state = F.relu(node_state)
            node_state = self.dropout(node_state)
            layer_graph_states.append(global_mean_pool(node_state, batch))
        return node_state, layer_graph_states


class LanguageEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        language_backbone: str = "huggingface",
        num_layers: int = 3,
        dropout: float = 0.3,
        use_language: bool = True,
        language_model_name: str = "DeepChem/ChemBERTa-77M-MLM",
        freeze_language_backbone: bool = True,
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.language_backbone = language_backbone.lower()
        self.num_layers = num_layers
        self.use_language = use_language and self.language_backbone != "none"
        self.language_model_name = language_model_name
        self.freeze_language_backbone = freeze_language_backbone

        if self.language_backbone not in VALID_LANGUAGE_BACKBONES:
            raise ValueError(f"language_backbone must be one of {sorted(VALID_LANGUAGE_BACKBONES)}")

        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)  # only for precomputed-tensor inputs
        self.text_tokenizer = None
        self.text_model = None
        self.text_proj = None

        if self.use_language:
            try:
                from transformers import AutoModel
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("transformers is required for the language branch") from exc

            self.text_tokenizer = _load_text_tokenizer(self.language_model_name, trust_remote_code)
            self.text_model = AutoModel.from_pretrained(self.language_model_name, trust_remote_code=trust_remote_code)
            text_hidden = int(getattr(self.text_model.config, "hidden_size", hidden_dim))
            self.text_proj = nn.Linear(text_hidden, hidden_dim)
            if self.freeze_language_backbone:
                for parameter in self.text_model.parameters():
                    parameter.requires_grad = False

            for _ in range(num_layers):
                self.layers.append(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.LayerNorm(hidden_dim),
                        nn.Dropout(dropout),
                    )
                )
            if self.freeze_language_backbone and self.text_model is not None:
                self.text_model.eval()

    def train(self, mode: bool = True):
        """A frozen text backbone must never re-enter train() (dropout / norm stats)."""
        super().train(mode)
        if self.freeze_language_backbone and self.text_model is not None:
            self.text_model.eval()
        return self

    def _pool_text_backbone(self, lang: Any, device: torch.device) -> torch.Tensor:
        if isinstance(lang, torch.Tensor):
            return self.input_proj(lang.view(lang.size(0), -1).float())

        if isinstance(lang, str):
            texts = [lang]
        elif isinstance(lang, (list, tuple)):
            texts = ["" if item is None else str(item) for item in lang]
        else:
            texts = ["" if lang is None else str(lang)]

        if self.text_tokenizer is None or self.text_model is None or self.text_proj is None:
            raise RuntimeError("Text encoder is not initialized")

        encoded = self.text_tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = self.text_model(**encoded)
        hidden_states = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1).type_as(hidden_states)
        pooled = (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp_min(1.0)
        return self.text_proj(pooled)

    def forward(self, lang: Optional[Any], batch_size: int, device: torch.device) -> torch.Tensor:
        if not self.use_language or lang is None:
            return torch.zeros(batch_size, self.hidden_dim, device=device)
        lang_state = self._pool_text_backbone(lang, device)
        for layer in self.layers:
            lang_state = layer(lang_state)
            lang_state = self.dropout(lang_state)
        return lang_state
