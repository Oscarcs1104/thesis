"""Dataset splitting utilities.

`scaffold_split` reproduces DeepChem's deterministic Bemis-Murcko scaffold split
(largest scaffold groups first go to train, then val, then test) using RDKit only
-- deepchem is not a dependency here. `random_split_subsets` keeps the old
behaviour available behind a flag.

Both return three `torch.utils.data.Subset` objects so downstream loaders are
unchanged.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import List, Optional, Sequence, Tuple

from torch.utils.data import Subset

try:  # RDKit is already a hard dependency of the data pipeline
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
except Exception as exc:  # pragma: no cover - dependency guard
    raise RuntimeError("RDKit is required for scaffold splitting") from exc


def _generate_scaffold(smiles: str, include_chirality: bool = False) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    except Exception:
        return ""


def _extract_smiles(dataset, smiles_list: Optional[Sequence[str]]) -> List[str]:
    if smiles_list is not None:
        if len(smiles_list) != len(dataset):
            raise ValueError(f"smiles_list length {len(smiles_list)} != dataset length {len(dataset)}")
        return [str(s) for s in smiles_list]
    out: List[str] = []
    for i in range(len(dataset)):
        item = dataset[i]
        smi = getattr(item, "smiles", None)
        if smi is None:
            raise ValueError(
                "scaffold_split needs a SMILES per item: pass smiles_list=... or use graphs that carry .smiles"
            )
        out.append(str(smi))
    return out


def scaffold_split(
    dataset,
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
    seed: int = 2025,
    smiles_list: Optional[Sequence[str]] = None,
    include_chirality: bool = False,
) -> Tuple[Subset, Subset, Subset]:
    """Deterministic Bemis-Murcko scaffold split.

    `seed` only shuffles the order of *equally sized* scaffold groups, so the
    split is stable but a different seed still yields a slightly different
    train/val/test partition (useful for multi-seed error bars).
    """
    total = frac_train + frac_val + frac_test
    if abs(total - 1.0) > 1e-4:
        raise ValueError("fractions must sum to 1.0")

    smiles = _extract_smiles(dataset, smiles_list)

    groups: dict[str, List[int]] = defaultdict(list)
    for idx, smi in enumerate(smiles):
        groups[_generate_scaffold(smi, include_chirality)].append(idx)

    rng = random.Random(seed)
    # Sort by group size (desc); shuffle ties so the seed has an effect.
    group_sets = list(groups.values())
    rng.shuffle(group_sets)
    group_sets.sort(key=len, reverse=True)

    n_total = len(smiles)
    n_train_cutoff = frac_train * n_total
    n_val_cutoff = (frac_train + frac_val) * n_total

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    for group in group_sets:
        if len(train_idx) + len(group) <= n_train_cutoff:
            train_idx.extend(group)
        elif len(train_idx) + len(val_idx) + len(group) <= n_val_cutoff:
            val_idx.extend(group)
        else:
            test_idx.extend(group)

    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)


def random_split_subsets(
    dataset,
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
    seed: int = 2025,
) -> Tuple[Subset, Subset, Subset]:
    total = frac_train + frac_val + frac_test
    if abs(total - 1.0) > 1e-4:
        raise ValueError("fractions must sum to 1.0")
    n_total = len(dataset)
    indices = list(range(n_total))
    random.Random(seed).shuffle(indices)
    n_train = int(n_total * frac_train)
    n_val = int(n_total * frac_val)
    return (
        Subset(dataset, indices[:n_train]),
        Subset(dataset, indices[n_train : n_train + n_val]),
        Subset(dataset, indices[n_train + n_val :]),
    )


def split_dataset(
    dataset,
    strategy: str,
    frac_train: float,
    frac_val: float,
    frac_test: float,
    seed: int,
    smiles_list: Optional[Sequence[str]] = None,
) -> Tuple[Subset, Subset, Subset]:
    strategy = strategy.lower()
    if strategy == "scaffold":
        return scaffold_split(dataset, frac_train, frac_val, frac_test, seed, smiles_list=smiles_list)
    if strategy == "random":
        return random_split_subsets(dataset, frac_train, frac_val, frac_test, seed)
    raise ValueError(f"Unknown split strategy: {strategy!r} (expected 'scaffold' or 'random')")
