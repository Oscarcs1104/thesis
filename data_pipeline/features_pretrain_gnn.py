"""Featurization for the Hu et al. 2020 pretrained GIN ("Strategies for Pre-training GNNs").

Those checkpoints were trained on a 2-column atom / 2-column bond schema, which is NOT
our OGB 9+3 schema (data_pipeline/features.py). Loading pretrained weights and feeding
them differently-shaped indices would silently mean nothing, so this schema has to be
reproduced exactly as chem/loader.py defines it:

    x         [N, 2]  [atomic_num - 1, chirality index]
    edge_attr [E, 2]  [bond type index, bond direction index]

Verified against snap-stanford/pretrain-gnns/chem/model_gin/contextpred.pth:
x_embedding1 (120, 300), x_embedding2 (3, 300), edge_embedding1 (6, 300),
edge_embedding2 (3, 300), 5 layers, emb_dim 300.

Note the chirality trap: the upstream list has FOUR entries (UNSPECIFIED, CW, CCW, OTHER)
while num_chirality_tag is 3, so a CHI_OTHER atom indexes past the embedding. Upstream
would crash or read garbage; we clamp, and count how often it happens.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data

FEATURE_VERSION = "pretrain-gnn-v1"

NUM_ATOM_TYPE = 120       # 118 elements + 2 mask tokens
NUM_CHIRALITY_TAG = 3
NUM_BOND_TYPE = 6         # 4 real + self-loop + mask
NUM_BOND_DIRECTION = 3

_CHIRALITY = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
_BONDS = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
_BOND_DIRS = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
]


def _safe_index(choices: list, value, hi: int) -> int:
    try:
        idx = choices.index(value)
    except ValueError:
        idx = 0
    return min(idx, hi)


def smiles_to_data_pretrain(smiles: str, target: Optional[float] = None) -> Optional[Data]:
    """SMILES -> PyG Data in the Hu et al. schema. None if RDKit rejects the string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    atoms = [
        [
            min(max(atom.GetAtomicNum() - 1, 0), NUM_ATOM_TYPE - 1),
            _safe_index(_CHIRALITY, atom.GetChiralTag(), NUM_CHIRALITY_TAG - 1),
        ]
        for atom in mol.GetAtoms()
    ]
    x = torch.tensor(np.asarray(atoms), dtype=torch.long)

    edges, edge_feats = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = [
            _safe_index(_BONDS, bond.GetBondType(), NUM_BOND_TYPE - 1),
            _safe_index(_BOND_DIRS, bond.GetBondDir(), NUM_BOND_DIRECTION - 1),
        ]
        edges += [[i, j], [j, i]]
        edge_feats += [feat, feat]

    if edges:
        edge_index = torch.tensor(np.asarray(edges, dtype=np.int64).T, dtype=torch.long)
        edge_attr = torch.tensor(np.asarray(edge_feats), dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.smiles = Chem.MolToSmiles(mol)
    if target is not None:
        data.y = torch.tensor([float(target)], dtype=torch.float)
    return data


__all__ = ["smiles_to_data_pretrain", "FEATURE_VERSION", "NUM_ATOM_TYPE",
           "NUM_CHIRALITY_TAG", "NUM_BOND_TYPE", "NUM_BOND_DIRECTION"]
