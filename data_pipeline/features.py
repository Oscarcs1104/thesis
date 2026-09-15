"""OGB-style molecular graph featurization (integer feature indices).

Every atom becomes a length-9 vector of *category indices* and every bond a
length-3 vector, exactly like `ogb.utils.features.atom_to_feature_vector` /
`bond_to_feature_vector` -- but implemented here so `ogb` is not a dependency.
The model side (`model/atom_bond_encoders.py`) turns these indices into learned
embeddings (one `nn.Embedding` per column, summed), which is the standard input
for GIN-E / GCN on molecular graphs and is what Chemprop / OGB leaderboards use.

`FEATURE_VERSION` is bumped whenever the encoding changes so cached `*.graphs`
tensors on disk are invalidated instead of silently reused with the wrong schema.
"""
from __future__ import annotations

from typing import List, Optional

import torch
from rdkit import Chem
from torch_geometric.data import Data

FEATURE_VERSION = "ogb-v1"

# --------------------------------------------------------------------------- #
# Allowed categories. The last slot of every list is a catch-all ("misc").
# --------------------------------------------------------------------------- #
_ATOM_FEATURES = {
    "atomic_num": list(range(1, 119)) + ["misc"],
    "chirality": ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER", "misc"],
    "degree": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"],
    "formal_charge": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, "misc"],
    "num_hs": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
    "num_radical_electrons": [0, 1, 2, 3, 4, "misc"],
    "hybridization": ["SP", "SP2", "SP3", "SP3D", "SP3D2", "misc"],
    "is_aromatic": [False, True],
    "is_in_ring": [False, True],
}
_BOND_FEATURES = {
    "bond_type": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc"],
    "stereo": ["STEREONONE", "STEREOZ", "STEREOE", "STEREOCIS", "STEREOTRANS", "STEREOANY"],
    "is_conjugated": [False, True],
}

ATOM_FEATURE_NAMES: List[str] = list(_ATOM_FEATURES)
BOND_FEATURE_NAMES: List[str] = list(_BOND_FEATURES)
ATOM_FEATURE_DIMS: List[int] = [len(v) for v in _ATOM_FEATURES.values()]
BOND_FEATURE_DIMS: List[int] = [len(v) for v in _BOND_FEATURES.values()]


def _safe_index(choices: list, value) -> int:
    try:
        return choices.index(value)
    except ValueError:
        return len(choices) - 1  # "misc" bucket


def atom_to_feature_vector(atom: Chem.Atom) -> List[int]:
    return [
        _safe_index(_ATOM_FEATURES["atomic_num"], atom.GetAtomicNum()),
        _safe_index(_ATOM_FEATURES["chirality"], str(atom.GetChiralTag())),
        _safe_index(_ATOM_FEATURES["degree"], atom.GetTotalDegree()),
        _safe_index(_ATOM_FEATURES["formal_charge"], atom.GetFormalCharge()),
        _safe_index(_ATOM_FEATURES["num_hs"], atom.GetTotalNumHs()),
        _safe_index(_ATOM_FEATURES["num_radical_electrons"], atom.GetNumRadicalElectrons()),
        _safe_index(_ATOM_FEATURES["hybridization"], str(atom.GetHybridization())),
        _safe_index(_ATOM_FEATURES["is_aromatic"], atom.GetIsAromatic()),
        _safe_index(_ATOM_FEATURES["is_in_ring"], atom.IsInRing()),
    ]


def bond_to_feature_vector(bond: Chem.Bond) -> List[int]:
    return [
        _safe_index(_BOND_FEATURES["bond_type"], str(bond.GetBondType())),
        _safe_index(_BOND_FEATURES["stereo"], str(bond.GetStereo())),
        _safe_index(_BOND_FEATURES["is_conjugated"], bond.GetIsConjugated()),
    ]


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Canonical SMILES, or None if RDKit cannot parse it / it has no atoms."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(mol)


def randomize_smiles(smiles: str) -> str:
    """A randomized (non-canonical) SMILES for the same molecule; input unchanged on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol, canonical=False, doRandom=True)


def mol_to_graph(mol: Chem.Mol, target: Optional[float] = None) -> Data:
    x = torch.tensor([atom_to_feature_vector(a) for a in mol.GetAtoms()], dtype=torch.long)

    src, dst, edge_feats = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = bond_to_feature_vector(bond)
        src += [i, j]
        dst += [j, i]
        edge_feats += [feat, feat]

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_feats, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, len(BOND_FEATURE_DIMS)), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if target is not None and target != "":
        try:
            data.y = torch.tensor([float(target)], dtype=torch.float)
        except (TypeError, ValueError):
            pass
    data.smiles = Chem.MolToSmiles(mol)  # store canonical form for every consumer
    data.feature_version = FEATURE_VERSION
    return data


def smiles_to_data(smiles: str, target: Optional[float] = None) -> Optional[Data]:
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return mol_to_graph(mol, target)
