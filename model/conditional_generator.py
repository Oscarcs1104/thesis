"""Standalone property-conditioned molecule generator (SELFIES).

Learns ``p(molecule | target property bin)`` -- the conditioning is the *target
property value*, discretized into quantile bins, prepended as a learned token.
It does NOT see the graph/text encoders or any input molecule (that was the flaw
in the old joint decoder: conditioning on the input molecule's own encoding made
the property signal redundant). Decoupled from the predictor on purpose; the
predictor is only used, separately, to (a) pseudo-label ZINC for fine-tuning and
(b) score generated molecules for the MAD metric. See docs/evaluation.md.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from model.smiles_decoder import SmilesDecoder


class PropertyBinner:
    """Quantile bins over a 1-D property. Bin edges are fit on a reference sample
    (the real dataset's TRAIN targets) so the bins line up with what evaluation
    conditions on. ``num_bins`` real bins + one 'unconditional' slot at index
    ``num_bins`` used for unconditional pretraining / classifier-free guidance.
    """

    def __init__(self, edges: List[float], name: str = "property") -> None:
        self.edges = list(map(float, edges))          # inner edges, length num_bins - 1
        self.num_bins = len(self.edges) + 1
        self.null_bin = self.num_bins                 # index for "no condition"
        self.name = name

    @classmethod
    def fit(cls, values, num_bins: int = 10, name: str = "property") -> "PropertyBinner":
        v = np.asarray(list(values), dtype=np.float64)
        v = v[np.isfinite(v)]
        qs = np.linspace(0, 1, num_bins + 1)[1:-1]
        edges = list(np.quantile(v, qs)) if v.size else list(np.linspace(-1, 1, num_bins - 1))
        return cls(edges, name=name)

    def to_bin(self, value: float) -> int:
        return int(np.searchsorted(self.edges, float(value), side="right"))

    def to_bins(self, values) -> np.ndarray:
        return np.searchsorted(self.edges, np.asarray(values, dtype=np.float64), side="right").astype(np.int64)

    def bin_center(self, bin_idx: int) -> float:
        lo = self.edges[bin_idx - 1] if bin_idx > 0 else self.edges[0] - (self.edges[1] - self.edges[0] if len(self.edges) > 1 else 1.0)
        hi = self.edges[bin_idx] if bin_idx < len(self.edges) else self.edges[-1] + (self.edges[-1] - self.edges[-2] if len(self.edges) > 1 else 1.0)
        return 0.5 * (lo + hi)

    def bin_edges(self, bin_idx: int) -> tuple:
        lo = self.edges[bin_idx - 1] if bin_idx > 0 else float("-inf")
        hi = self.edges[bin_idx] if bin_idx < len(self.edges) else float("inf")
        return lo, hi

    def state_dict(self) -> dict:
        return {"edges": self.edges, "name": self.name}

    @classmethod
    def from_state_dict(cls, state: dict) -> "PropertyBinner":
        return cls(state["edges"], name=state.get("name", "property"))


class ConditionalSmilesGenerator(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        pad_idx: int,
        start_idx: int,
        end_idx: int,
        num_property_bins: int,
        decoder_layers: int = 6,
        decoder_heads: int = 8,
        max_len: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_property_bins = num_property_bins
        self.null_bin_idx = num_property_bins
        # +1 slot = unconditional
        self.property_embedding = nn.Embedding(num_property_bins + 1, hidden_dim)
        self.decoder = SmilesDecoder(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            pad_idx=pad_idx,
            start_idx=start_idx,
            end_idx=end_idx,
            num_layers=decoder_layers,
            num_heads=decoder_heads,
            max_len=max_len,
            dropout=dropout,
        )

    def _latent(self, property_bins: torch.Tensor) -> torch.Tensor:
        return self.property_embedding(property_bins.long())

    def forward(self, property_bins: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        return self.decoder(self._latent(property_bins), input_ids)

    @torch.no_grad()
    def sample(
        self,
        property_bin: int,
        id_to_token: Dict[int, str],
        num_samples: int,
        max_len: int = 96,
        temperature: float = 1.0,
        device: Optional[torch.device] = None,
        chunk_size: int = 512,
        as_selfies: bool = False,
    ) -> List[str]:
        self.eval()
        device = device or next(self.parameters()).device
        out: List[str] = []
        remaining = num_samples
        while remaining > 0:
            b = min(chunk_size, remaining)
            bins = torch.full((b,), int(property_bin), dtype=torch.long, device=device)
            latent = self._latent(bins)
            out.extend(self.decoder.generate_batch(
                latent, id_to_token, max_len=max_len, temperature=temperature, sample=True, as_selfies=as_selfies
            ))
            remaining -= b
        return out
