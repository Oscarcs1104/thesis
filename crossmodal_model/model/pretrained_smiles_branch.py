"""Pretrained-transformer SMILES branch (ChemBERTa by default -- same backbone as
thesis_model's own LanguageEncoder) for HybridMoLA's fusion, instead of MoLA's
from-scratch char-level branch (see encoder_hybrid.py).

MoLA's cross-layer fusion expects ONE token per graph layer per modality (see
GraphEncoder.forward's layer_graph_states). A pretrained transformer only gives a single
pooled vector per molecule -- no notion of "layer l". Reused here exactly as
thesis_model.model.encoders.LanguageEncoder does it (frozen backbone + a stack of
trainable Linear->GELU->LayerNorm->Dropout blocks on top), except the OUTPUT of each
block is kept (not just the last), so each block's output stands in for "this modality's
token at graph-layer l" -- the same trick, just exposing every intermediate step.
"""
from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn

from thesis_model.model.encoders import _load_text_tokenizer


class PretrainedSmilesBranch(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        model_name: str = "DeepChem/ChemBERTa-77M-MLM",
        freeze_backbone: bool = True,
        trust_remote_code: bool = False,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.freeze_backbone = freeze_backbone
        self.model_name = model_name

        self.tokenizer = _load_text_tokenizer(model_name, trust_remote_code)
        self.text_model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        text_hidden = int(getattr(self.text_model.config, "hidden_size", hidden_dim))
        self.text_proj = nn.Linear(text_hidden, hidden_dim)
        if freeze_backbone:
            for p in self.text_model.parameters():
                p.requires_grad = False
            self.text_model.eval()

        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
            for _ in range(num_layers)
        ])

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.text_model.eval()
        return self

    def _pool(self, smiles_list: Sequence[str], device: torch.device) -> torch.Tensor:
        texts = ["" if s is None else str(s) for s in smiles_list]
        encoded = self.tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.set_grad_enabled(not self.freeze_backbone):
            outputs = self.text_model(**encoded)
        hidden_states = outputs.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).type_as(hidden_states)
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.text_proj(pooled)

    def forward(self, smiles_list: Sequence[str], device: torch.device) -> List[torch.Tensor]:
        """Returns num_layers tensors [B, hidden_dim], one per graph layer -- the i-th is
        the pooled+projected ChemBERTa embedding passed through i+1 trainable blocks."""
        h = self._pool(smiles_list, device)
        states = []
        for layer in self.layers:
            h = layer(h)
            states.append(h)
        return states
