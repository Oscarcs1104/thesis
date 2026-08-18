"""Shared 2D molecule metrics: fingerprints, Tanimoto similarity, Murcko scaffolds, Shannon entropy.

Used by tools/check_diversity.py and tools/demo_generate_property.py.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


def mols_from_smiles(smiles_list: Sequence[str]) -> Tuple[List[Chem.Mol], int]:
    """Parse a list of SMILES. Returns (valid_mols, num_invalid)."""
    mols: List[Chem.Mol] = []
    invalid = 0
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            invalid += 1
        else:
            mols.append(mol)
    return mols, invalid


def morgan_fingerprints(mols: Sequence[Chem.Mol], radius: int = 2, n_bits: int = 2048):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return [generator.GetFingerprint(mol) for mol in mols]


def mean_pairwise_tanimoto(fps: Sequence) -> Optional[float]:
    """Mean pairwise Tanimoto similarity. Lower means more internally diverse."""
    n = len(fps)
    if n < 2:
        return None
    total = 0.0
    count = 0
    for i in range(n - 1):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        total += sum(sims)
        count += len(sims)
    return total / count if count else None


def nearest_neighbor_similarity(query_fps: Sequence, reference_fps: Sequence) -> List[float]:
    """For each query fingerprint, the highest Tanimoto similarity to any reference fingerprint."""
    if not reference_fps:
        return [0.0 for _ in query_fps]
    return [max(DataStructs.BulkTanimotoSimilarity(fp, reference_fps)) for fp in query_fps]


def murcko_scaffold_smiles(mol: Chem.Mol) -> str:
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold) if scaffold is not None else ""


def shannon_entropy(labels: Sequence[str]) -> float:
    """Shannon entropy (nats) of a label distribution -- e.g. scaffold SMILES per molecule."""
    if not labels:
        return 0.0
    counts: Dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    total = len(labels)
    return -sum((c / total) * math.log(c / total) for c in counts.values())
