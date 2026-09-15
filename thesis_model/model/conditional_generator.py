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

from common.property_bins import PropertyBinner  # re-exported: moved to common/, both halves use it
from thesis_model.model.smiles_decoder import SmilesDecoder


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


__all__ = ["ConditionalSmilesGenerator", "PropertyBinner"]
