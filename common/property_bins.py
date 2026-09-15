"""Quantile binning for conditioning values, shared by both halves of the thesis.

Moved here from thesis_model/model/conditional_generator.py (which re-exports it)
because crossmodal_model's pair-conditioned generator needs the same binning, and a
utility used by both models should not live inside either one's package.

Why bins rather than feeding the raw scalar to a Linear(1 -> H): one number through a
linear layer is a weak, low-frequency signal, and the old code fed it unstandardized
(raw kcal/mol). A bin index selects a learned embedding, so every conditioning value
starts with a full-rank vector the decoder can actually use. This is the same problem
Fourier features solve; bins solve it too, so the two are alternatives, not a stack.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


class PropertyBinner:
    """Quantile bins over a 1-D property (or a 1-D *delta* between two molecules).

    Edges are fit on a reference sample so the bins line up with what evaluation
    conditions on. ``num_bins`` real bins plus one 'unconditional' slot at index
    ``num_bins``, used for condition dropout and classifier-free guidance.
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
        # Quantile edges collapse when the distribution is spiky (QED piles up at its
        # ceiling, deltas pile up at 0). Duplicated edges would make empty bins that no
        # value can ever land in, so the embedding rows would train on nothing.
        edges = sorted(set(edges))
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


class DeltaBinnerSet:
    """One PropertyBinner per conditioned property, kept together.

    Each property carries its own null bin and is dropped independently during
    training, so at sampling time you can ask for "delta logP = +1, don't care about
    the rest" -- which is the request a chemist actually makes. Dropping the whole
    prefix jointly would only ever allow all-or-nothing conditioning.
    """

    def __init__(self, binners: Dict[str, PropertyBinner]) -> None:
        self.binners = dict(binners)
        self.names = list(self.binners)

    @classmethod
    def fit(cls, deltas: np.ndarray, names: Sequence[str], num_bins: int = 20) -> "DeltaBinnerSet":
        if deltas.shape[1] != len(names):
            raise ValueError(f"deltas has {deltas.shape[1]} columns but {len(names)} names were given")
        return cls({n: PropertyBinner.fit(deltas[:, i], num_bins=num_bins, name=n)
                    for i, n in enumerate(names)})

    @property
    def vocab_sizes(self) -> List[int]:
        """Embedding rows per property: real bins + the null slot."""
        return [self.binners[n].num_bins + 1 for n in self.names]

    def to_bins(self, deltas: np.ndarray) -> np.ndarray:
        """[N, P] float deltas -> [N, P] int64 bin indices."""
        return np.stack([self.binners[n].to_bins(deltas[:, i]) for i, n in enumerate(self.names)], axis=1)

    def null_bins(self) -> List[int]:
        return [self.binners[n].null_bin for n in self.names]

    def state_dict(self) -> dict:
        return {"names": self.names, "binners": {n: b.state_dict() for n, b in self.binners.items()}}

    @classmethod
    def from_state_dict(cls, state: dict) -> "DeltaBinnerSet":
        binners = {n: PropertyBinner.from_state_dict(state["binners"][n]) for n in state["names"]}
        out = cls(binners)
        out.names = list(state["names"])  # preserve column order, dict order is not authoritative
        return out


__all__ = ["PropertyBinner", "DeltaBinnerSet"]
