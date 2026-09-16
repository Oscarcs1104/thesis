"""Curation steps applied before any molecule enters the pipeline.

Two things the rest of the pipeline did not handle: disconnected fragments and target
outliers. They are treated very differently on purpose, and the asymmetry is the point.

DISCONNECTED FRAGMENTS are removed. A SMILES containing a dot is more than one species --
typically a parent plus a counterion (a hydrochloride, a sodium salt) or a solvate. The
measured property is attributed to the parent, and the counterion contributes atoms the
model must then learn to ignore. Every published MoleculeNet protocol strips them, so
keeping them would make the numbers incomparable as well as noisier.

OUTLIERS are DETECTED AND REPORTED, NOT REMOVED. Dropping extreme targets from a standard
benchmark silently changes the benchmark: RMSE falls because the hardest molecules are
gone, and the result can no longer be compared against any published figure on ESOL,
FreeSolv or Lipophilicity. An extreme hydration free energy is a real measurement of a
real compound, not a data-entry error, and a model that cannot predict it is a model with
a limitation worth reporting. The count and identity of the extremes are recorded so the
write-up can state them; the split keeps them.

If a supervisor requires removal, --drop-outliers does it, and the report says so, so a
table produced that way is never mistaken for one that is comparable to the literature.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def largest_fragment(smiles: str) -> Optional[str]:
    """Canonical SMILES of the largest fragment by heavy-atom count, or None if unparseable.

    Size rather than RDKit's LargestFragmentChooser default (which weighs by molecular
    weight) because a heavy counterion such as iodide can outweigh a small organic parent
    while obviously not being the compound the measurement refers to.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags:
        return None
    biggest = max(frags, key=lambda m: (m.GetNumHeavyAtoms(), Chem.MolToSmiles(m)))
    return Chem.MolToSmiles(biggest)


def count_fragments(smiles: str) -> int:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    mol = Chem.MolFromSmiles(smiles)
    return 0 if mol is None else len(Chem.GetMolFrags(mol))


def strip_fragments(smiles_list: Sequence[str]) -> Tuple[List[Optional[str]], Dict[str, int]]:
    """Reduce every entry to its largest fragment. Returns (cleaned, report)."""
    cleaned: List[Optional[str]] = []
    n_multi = n_failed = 0
    for smi in smiles_list:
        n = count_fragments(smi)
        if n == 0:
            cleaned.append(None)
            n_failed += 1
            continue
        if n > 1:
            n_multi += 1
        cleaned.append(largest_fragment(smi))
    return cleaned, {
        "n_total": len(smiles_list),
        "n_multi_fragment": n_multi,
        "n_unparseable": n_failed,
    }


def find_outliers(values: Sequence[float], n_mad: float = 5.0) -> Dict[str, object]:
    """Flag extreme targets by the modified z-score (median absolute deviation).

    MAD rather than mean and standard deviation because the standard deviation is itself
    inflated by the very points being looked for, which makes a plain z-score miss them.
    n_mad = 5 is deliberately permissive: the aim is to notice genuinely extreme values,
    not to trim the tails of a legitimately wide distribution.
    """
    v = np.asarray(list(values), dtype=np.float64)
    finite = np.isfinite(v)
    med = float(np.median(v[finite])) if finite.any() else float("nan")
    mad = float(np.median(np.abs(v[finite] - med))) if finite.any() else 0.0
    # 0.6745 rescales the MAD so the score matches a z-score for normal data.
    scale = 1.4826 * mad
    if scale <= 0:
        scores = np.zeros_like(v)
    else:
        scores = np.abs(v - med) / scale
    mask = np.isfinite(v) & (scores > n_mad)
    return {
        "median": med,
        "mad": mad,
        "n_mad": n_mad,
        "n_outliers": int(mask.sum()),
        "fraction": float(mask.mean()) if v.size else 0.0,
        "indices": np.flatnonzero(mask).tolist(),
        "values": v[mask].tolist(),
        "policy": "reported, not removed",
    }


def curation_summary(report: Dict[str, int], outliers: Dict[str, object], name: str) -> str:
    lines = [f"[{name}] curation"]
    lines.append(f"  molecules              {report['n_total']:>8,}")
    if report.get("n_unparseable"):
        lines.append(f"  unparseable (dropped)  {report['n_unparseable']:>8,}")
    lines.append(f"  multi-fragment         {report['n_multi_fragment']:>8,}"
                 f"  -> reduced to largest fragment")
    if outliers:
        lines.append(f"  extreme targets        {outliers['n_outliers']:>8,}"
                     f"  ({outliers['fraction']:.2%}, >{outliers['n_mad']:g} MAD) -- {outliers['policy']}")
    return "\n".join(lines)


__all__ = ["largest_fragment", "count_fragments", "strip_fragments", "find_outliers",
           "curation_summary"]
