from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, GINConv, global_mean_pool

VALID_GRAPH_BACKBONES = {"gcn", "gat", "gatv2", "gin"}
VALID_LANGUAGE_BACKBONES = {"huggingface", "none"}


def _load_text_tokenizer(model_name: str, trust_remote_code: bool):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    except OSError:
        # Some trust_remote_code repos declare a custom tokenizer class (via auto_map)
        # whose source file doesn't actually exist upstream (seen with
        # DeepChem/MoLFormer-c3-1.1B, which points at a missing file in
        # ibm/MoLFormer-XL-both-10pct). Fall back to building a fast tokenizer
        # straight from the repo's own tokenizer.json, bypassing that broken lookup.
        import json

        from huggingface_hub import hf_hub_download
        from transformers import PreTrainedTokenizerFast

        tokenizer_file = hf_hub_download(model_name, "tokenizer.json")
        special_tokens = {}
        try:
            special_file = hf_hub_download(model_name, "special_tokens_map.json")
            with open(special_file, "r", encoding="utf-8") as handle:
                raw_special = json.load(handle)
            special_tokens = {key: (value["content"] if isinstance(value, dict) else value) for key, value in raw_special.items()}
        except Exception:
            pass
        return PreTrainedTokenizerFast(tokenizer_file=tokenizer_file, **special_tokens)


def _make_graph_conv(backbone: str, in_dim: int, out_dim: int) -> nn.Module:
    backbone = backbone.lower()
    if backbone == "gcn":
        return GCNConv(in_dim, out_dim)
    if backbone == "gat":
        return GATConv(in_dim, out_dim, heads=4, concat=False)
    if backbone == "gatv2":
        return GATv2Conv(in_dim, out_dim, heads=4, concat=False)
    if backbone == "gin":
        return GINConv(
            nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
            )
        )
    raise ValueError(f"Unsupported graph backbone: {backbone}")


class NodeFeatureEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        node_encoding: str = "categorical",
        node_vocab_sizes: Optional[Sequence[int]] = None,
        node_input_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_encoding = node_encoding.lower()
        self.node_input_dim = int(node_input_dim) if node_input_dim is not None else None

        if self.node_encoding not in {"categorical", "dense"}:
            raise ValueError("node_encoding must be 'categorical' or 'dense'")

        if self.node_encoding == "categorical":
            if not node_vocab_sizes:
                raise ValueError("node_vocab_sizes is required when node_encoding='categorical'")
            self.embeddings = nn.ModuleList([nn.Embedding(int(size), hidden_dim) for size in node_vocab_sizes])
            self.proj = None
        else:
            self.embeddings = None
            if self.node_input_dim is not None:
                self.proj = nn.Linear(self.node_input_dim, hidden_dim)
            else:
                self.proj = None

    def _ensure_dense_projection(self, x: torch.Tensor) -> nn.Linear:
        if self.proj is None:
            input_dim = int(x.size(-1)) if x.dim() >= 2 else 1
            self.proj = nn.Linear(input_dim, self.hidden_dim)
            self.proj = self.proj.to(x.device)
            return self.proj

        if self.proj.in_features != int(x.size(-1)):
            input_dim = int(x.size(-1)) if x.dim() >= 2 else 1
            self.proj = nn.Linear(input_dim, self.hidden_dim)
            self.proj = self.proj.to(x.device)
        return self.proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dense inputs go through one projection.
        if self.node_encoding == "dense":
            x = x.float()
            proj = self._ensure_dense_projection(x)
            return proj(x)

        # Categorical fields are embedded one by one and then summed.
        x = x.long()
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if x.size(-1) != len(self.embeddings):
            # If inputs are floats, treat them as dense continuous features and project.
            if x.is_floating_point():
                if not hasattr(self, "_fallback_proj") or self._fallback_proj is None:
                    input_dim = int(x.size(-1)) if x.dim() >= 2 else 1
                    self._fallback_proj = nn.Linear(input_dim, self.hidden_dim)
                self._fallback_proj = self._fallback_proj.to(x.device)
                return self._fallback_proj(x.float())

            # Otherwise warn once and fall back to dense projection.
            if not hasattr(self, "_warned") or not self._warned:
                warn_msg = (
                    f"NodeFeatureEncoder: expected {len(self.embeddings)} categorical fields, got {x.size(-1)}; "
                    "falling back to dense projection."
                )
                print(warn_msg)
                self._warned = True
            if not hasattr(self, "_fallback_proj") or self._fallback_proj is None:
                input_dim = int(x.size(-1)) if x.dim() >= 2 else 1
                self._fallback_proj = nn.Linear(input_dim, self.hidden_dim)
            self._fallback_proj = self._fallback_proj.to(x.device)
            return self._fallback_proj(x.float())

        encoded = 0
        for field_idx, embedding in enumerate(self.embeddings):
            encoded = encoded + embedding(x[:, field_idx])
        return encoded


class GraphEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        graph_backbone: str = "gatv2",
        num_layers: int = 3,
        dropout: float = 0.3,
        node_encoding: str = "categorical",
        node_vocab_sizes: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        # One encoder shared by all graph layers.
        self.node_encoder = NodeFeatureEncoder(
            hidden_dim=hidden_dim,
            node_encoding=node_encoding,
            node_vocab_sizes=node_vocab_sizes,
        )

        # Stack the requested number of graph convolutions.
        self.graph_layers = nn.ModuleList()
        self.graph_layers.append(_make_graph_conv(graph_backbone, hidden_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.graph_layers.append(_make_graph_conv(graph_backbone, hidden_dim, hidden_dim))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # Pool after every layer; the model only consumes the final one for now.
        node_state = self.node_encoder(x)
        layer_graph_states: List[torch.Tensor] = []

        for graph_layer in self.graph_layers:
            node_state = graph_layer(node_state, edge_index)
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
        # Only exercised when `lang` arrives as a precomputed tensor instead of raw text.
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)
        self.text_tokenizer = None
        self.text_model = None
        self.text_proj = None

        if self.use_language:
            try:
                from transformers import AutoModel
            except Exception as exc:  # pragma: no cover - runtime dependency guard
                raise RuntimeError(
                    "transformers is required for the language branch. Install transformers and sentencepiece."
                ) from exc

            self.text_tokenizer = _load_text_tokenizer(self.language_model_name, trust_remote_code)
            self.text_model = AutoModel.from_pretrained(self.language_model_name, trust_remote_code=trust_remote_code)
            text_hidden = int(getattr(self.text_model.config, "hidden_size", hidden_dim))
            self.text_proj = nn.Linear(text_hidden, hidden_dim)
            if self.freeze_language_backbone:
                for parameter in self.text_model.parameters():
                    parameter.requires_grad = False

            for layer_idx in range(num_layers):
                self.layers.append(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.LayerNorm(hidden_dim),
                        nn.Dropout(dropout),
                    )
                )

    def _pool_text_backbone(self, lang: Any, device: torch.device) -> torch.Tensor:
        # The HF text encoder works on raw SMILES strings or precomputed tensors.
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

        encoded = self.text_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
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
