from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, GINConv, GINEConv, global_add_pool, global_max_pool, global_mean_pool

try:
    from data_pipeline.convert_smiles_to_pyg import ATOM_VOCAB_SIZES, BOND_VOCAB_SIZES
except Exception:  # pragma: no cover - import-path guard when loaded standalone
    ATOM_VOCAB_SIZES = [119, 11, 7, 9, 8, 2, 2, 4]
    BOND_VOCAB_SIZES = [22, 2, 2, 6]

VALID_GRAPH_BACKBONES = {"gcn", "gat", "gatv2", "gin"}
VALID_LANGUAGE_BACKBONES = {"huggingface", "none"}
# D5: which backbones actually consume edge_attr, and how.
# - gin: swapped for GINEConv, which adds a (projected) edge embedding to each message.
# - gat/gatv2: PyG's native `edge_dim` arg folds edge features into the attention score.
# - gcn: GCNConv only accepts a scalar edge_weight, not a multi-dim edge feature -- no
#   edge-aware variant here, it silently keeps behaving exactly as before.
EDGE_AWARE_BACKBONES = {"gin", "gat", "gatv2"}


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


def _make_graph_conv(backbone: str, in_dim: int, out_dim: int, edge_dim: Optional[int] = None) -> nn.Module:
    backbone = backbone.lower()
    if backbone == "gcn":
        return GCNConv(in_dim, out_dim)
    if backbone == "gat":
        return GATConv(in_dim, out_dim, heads=4, concat=False, edge_dim=edge_dim)
    if backbone == "gatv2":
        return GATv2Conv(in_dim, out_dim, heads=4, concat=False, edge_dim=edge_dim)
    if backbone == "gin":
        mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        if edge_dim is not None:
            # D5: GINEConv adds a (linearly projected, edge_dim -> in_dim) edge
            # embedding to each source node's message before the MLP, instead of
            # GINConv's plain sum-of-neighbors that's blind to bond type/ring/etc.
            return GINEConv(mlp, edge_dim=edge_dim)
        return GINConv(mlp)
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
            # D5: default to the full 8-field OGB-style atom vocab (see
            # data_pipeline/convert_smiles_to_pyg.ATOM_VOCAB_SIZES) rather than
            # requiring every caller to spell it out; still overridable.
            vocab_sizes = list(node_vocab_sizes) if node_vocab_sizes else list(ATOM_VOCAB_SIZES)
            self.embeddings = nn.ModuleList([nn.Embedding(int(size), hidden_dim) for size in vocab_sizes])
            self.proj = None
            self.input_norm = None
        else:
            self.embeddings = None
            # Dense atom features are always the fixed 7-dim vector from
            # convert_smiles_to_pyg.atom_features() unless the caller overrides
            # node_input_dim. Create proj/input_norm eagerly here (not lazily on
            # first forward) so they're registered submodules *before*
            # load_state_dict() runs at inference time -- otherwise a freshly
            # loaded checkpoint's trained input_norm (weight/bias/running stats)
            # is silently dropped (the key doesn't exist yet, strict=False hides
            # it), and a lazily-created BatchNorm also never inherits
            # model.eval(), so it uses batch statistics at inference and crashes
            # outright on a batch of size 1 (e.g. a single-atom molecule alone).
            resolved_input_dim = self.node_input_dim if self.node_input_dim is not None else 7
            self.proj = nn.Linear(resolved_input_dim, hidden_dim)
            self.input_norm = nn.BatchNorm1d(resolved_input_dim)

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

    def _ensure_input_norm(self, x: torch.Tensor) -> nn.BatchNorm1d:
        input_dim = int(x.size(-1)) if x.dim() >= 2 else 1
        if self.input_norm is None or self.input_norm.num_features != input_dim:
            self.input_norm = nn.BatchNorm1d(input_dim).to(x.device)
            # A module created here (fallback path, mismatched dim) is fresh and
            # defaults to train() regardless of the parent's current mode -- sync
            # it so eval-mode inference doesn't silently use batch statistics.
            self.input_norm.train(self.training)
        return self.input_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dense inputs go through one projection.
        if self.node_encoding == "dense":
            x = x.float()
            # Raw atom features mix wildly different scales (atomic_num up to ~118,
            # formal_charge/aromatic in {0,1}, ...) -- normalize before the linear
            # projection so no single feature dominates the gradient.
            input_norm = self._ensure_input_norm(x)
            proj = self._ensure_dense_projection(x)
            if x.size(0) <= 1:
                # BatchNorm1d can't compute batch statistics from a single row (raises
                # "Expected more than 1 value per channel when training"). This is a
                # real case here, not just a theoretical one: a PyG-batched graph has
                # one row per atom, and a single-heavy-atom molecule (e.g. "C"/"N"/"S",
                # all present in data/freesolv.csv) landing alone in an undersized
                # trailing batch triggers it mid-training. Fall back to running
                # statistics (eval-mode behavior) for just this call instead of crashing.
                was_training = input_norm.training
                input_norm.eval()
                try:
                    normalized = input_norm(x)
                finally:
                    input_norm.train(was_training)
            else:
                normalized = input_norm(x)
            return proj(normalized)

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


class EdgeFeatureEncoder(nn.Module):
    """D5: embeds categorical bond features (see convert_smiles_to_pyg.bond_features_categorical)
    into hidden_dim, one nn.Embedding per field, summed -- mirrors NodeFeatureEncoder's
    categorical path so atom and bond embeddings live in comparable spaces."""

    def __init__(self, hidden_dim: int, edge_vocab_sizes: Optional[Sequence[int]] = None) -> None:
        super().__init__()
        vocab_sizes = list(edge_vocab_sizes) if edge_vocab_sizes else list(BOND_VOCAB_SIZES)
        self.embeddings = nn.ModuleList([nn.Embedding(int(size), hidden_dim) for size in vocab_sizes])

    def forward(self, edge_attr: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if edge_attr is None or edge_attr.numel() == 0:
            return None
        edge_attr = edge_attr.long()
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        if edge_attr.size(-1) != len(self.embeddings):
            return None  # shape mismatch (e.g. stale cache) -- degrade to edge-blind rather than crash
        encoded = 0
        for field_idx, embedding in enumerate(self.embeddings):
            encoded = encoded + embedding(edge_attr[:, field_idx])
        return encoded


_POOL_FNS = {"mean": global_mean_pool, "add": global_add_pool, "sum": global_add_pool, "max": global_max_pool}


class GraphEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        graph_backbone: str = "gatv2",
        num_layers: int = 3,
        dropout: float = 0.3,
        node_encoding: str = "categorical",
        node_vocab_sizes: Optional[Sequence[int]] = None,
        edge_vocab_sizes: Optional[Sequence[int]] = None,
        pool: str = "add",
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.graph_backbone = graph_backbone.lower()
        # One encoder shared by all graph layers.
        self.node_encoder = NodeFeatureEncoder(
            hidden_dim=hidden_dim,
            node_encoding=node_encoding,
            node_vocab_sizes=node_vocab_sizes,
        )

        # D5: edges only feed backbones that know what to do with them (see
        # EDGE_AWARE_BACKBONES); GCNConv keeps its pre-D5, edge-blind behavior.
        self.use_edge_attr = self.graph_backbone in EDGE_AWARE_BACKBONES
        self.edge_encoder = EdgeFeatureEncoder(hidden_dim, edge_vocab_sizes) if self.use_edge_attr else None
        edge_dim = hidden_dim if self.use_edge_attr else None

        # Stack the requested number of graph convolutions.
        self.graph_layers = nn.ModuleList()
        self.graph_layers.append(_make_graph_conv(graph_backbone, hidden_dim, hidden_dim, edge_dim=edge_dim))
        for _ in range(num_layers - 1):
            self.graph_layers.append(_make_graph_conv(graph_backbone, hidden_dim, hidden_dim, edge_dim=edge_dim))

        pool = pool.lower()
        if pool not in _POOL_FNS and pool != "mean_max":
            raise ValueError(f"pool must be one of {sorted(_POOL_FNS)} or 'mean_max', got {pool!r}")
        self.pool = pool
        # mean_max concatenates two hidden_dim pooled vectors then projects back down to
        # hidden_dim, so every downstream consumer (fusion head, MoE, cross-attention, ...)
        # keeps seeing exactly hidden_dim from the graph branch -- no cascading dim changes.
        self.pool_proj = nn.Linear(hidden_dim * 2, hidden_dim) if pool == "mean_max" else None

    def _pool(self, node_state: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.pool == "mean_max":
            pooled = torch.cat([global_mean_pool(node_state, batch), global_max_pool(node_state, batch)], dim=-1)
            return self.pool_proj(pooled)
        return _POOL_FNS[self.pool](node_state, batch)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # Pool after every layer; the model only consumes the final one for now.
        node_state = self.node_encoder(x)
        edge_state = self.edge_encoder(edge_attr) if self.edge_encoder is not None else None
        layer_graph_states: List[torch.Tensor] = []

        for graph_layer in self.graph_layers:
            if edge_state is not None:
                node_state = graph_layer(node_state, edge_index, edge_attr=edge_state)
            else:
                node_state = graph_layer(node_state, edge_index)
            node_state = F.relu(node_state)
            node_state = self.dropout(node_state)
            layer_graph_states.append(self._pool(node_state, batch))

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
            if self.freeze_language_backbone and self.text_model is not None:
                self.text_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_language_backbone and self.text_model is not None:
            self.text_model.eval()
        return self

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
