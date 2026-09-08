"""Multimodal property predictor: graph encoder + language encoder -> concat -> MLP head.

Predictor only. Molecule *generation* is a separate, standalone model
(`model/conditional_generator.py`) trained by `training/train_generator.py`; it is
deliberately decoupled from this predictor so the two contributions can be
evaluated independently (see docs/evaluation.md).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from model.encoders import VALID_GRAPH_BACKBONES, GraphEncoder, LanguageEncoder


class MultimodalModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        graph_backbone: str = "gin",
        language_backbone: str = "huggingface",
        num_layers: int = 3,
        dropout: float = 0.3,
        use_graph: bool = True,
        use_language: bool = True,
        language_model_name: str = "DeepChem/ChemBERTa-77M-MLM",
        freeze_language_backbone: bool = True,
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__()

        graph_backbone = graph_backbone.lower()
        if graph_backbone not in VALID_GRAPH_BACKBONES:
            raise ValueError(f"graph_backbone must be one of {sorted(VALID_GRAPH_BACKBONES)}")

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.graph_backbone = graph_backbone
        self.language_backbone = language_backbone.lower()
        self.num_layers = num_layers
        self.use_graph = use_graph
        self.use_language = use_language and self.language_backbone != "none"
        if not self.use_graph and not self.use_language:
            raise ValueError("At least one of use_graph / use_language must be enabled")
        self.dropout = nn.Dropout(dropout)

        self.graph_encoder = GraphEncoder(
            hidden_dim=hidden_dim,
            graph_backbone=graph_backbone,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.language_encoder = LanguageEncoder(
            hidden_dim=hidden_dim,
            language_backbone=self.language_backbone,
            num_layers=num_layers,
            dropout=dropout,
            use_language=self.use_language,
            language_model_name=language_model_name,
            freeze_language_backbone=freeze_language_backbone,
            trust_remote_code=trust_remote_code,
        )

        fused_dim = hidden_dim * (int(self.use_graph) + int(self.use_language))
        self.head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, output_dim),
        )

    def train(self, mode: bool = True):
        """Belt-and-suspenders: keep a frozen HF text backbone in eval()."""
        super().train(mode)
        text_model = getattr(self.language_encoder, "text_model", None)
        if getattr(self.language_encoder, "freeze_language_backbone", False) and text_model is not None:
            text_model.eval()
        return self

    def _get_states(self, data) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        batch = data.batch
        batch_size = int(batch.max().item()) + 1
        graph_state = None
        if self.use_graph:
            _, layer_graph_states = self.graph_encoder(
                data.x, data.edge_index, getattr(data, "edge_attr", None), batch
            )
            graph_state = layer_graph_states[-1]
        lang_state = self.language_encoder(getattr(data, "smiles", None), batch_size=batch_size, device=batch.device)
        return graph_state, lang_state

    def encode(self, data) -> torch.Tensor:
        graph_state, lang_state = self._get_states(data)
        if self.use_graph and self.use_language:
            return torch.cat([graph_state, lang_state], dim=-1)
        return graph_state if self.use_graph else lang_state

    def forward(self, data) -> torch.Tensor:
        return self.head(self.encode(data))


def build_model_from_args(args) -> MultimodalModel:
    return MultimodalModel(
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        graph_backbone=args.graph_backbone,
        language_backbone=args.language_backbone,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_graph=getattr(args, "use_graph", True),
        use_language=args.use_language,
        language_model_name=getattr(args, "language_model_name", "DeepChem/ChemBERTa-77M-MLM"),
        freeze_language_backbone=getattr(args, "freeze_language_backbone", True),
        trust_remote_code=getattr(args, "trust_remote_code", False),
    )
