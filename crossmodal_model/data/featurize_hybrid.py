"""Data prep for MoLA-Hybrid's graph branch: categorical atom/bond features
(data_pipeline.convert_smiles_to_pyg.smiles_to_data) instead of DeepChem's dense
features. SMILES branch keeps MoLA's own char-index tokenization (build_vocab).

Side benefit: keeps single-heavy-atom molecules that DeepChem's featurizer would drop,
so split sizes match thesis_model's exactly.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import torch

from data_pipeline.convert_smiles_to_pyg import smiles_to_data
from crossmodal_model.data.featurize import build_vocab  # noqa: F401 -- re-exported for callers


def prepare_hybrid_data(smiles_list: Sequence[str], y_list: Sequence[float], char_vocab: Dict[str, int], max_sm_len: int = 100) -> List["torch_geometric.data.Data"]:  # noqa: F821
    data_list = []
    for smi, y in zip(smiles_list, y_list):
        d = smiles_to_data(smi, target=float(y))
        if d is None:
            continue
        # SMILES branch tokenizes the RAW csv text (not smiles_to_data's re-canonicalized
        # form) -- same convention as crossmodal_model.data.featurize.prepare_data
        # elsewhere, so the char vocab built over the raw CSVs still applies unchanged.
        sm_idx = [char_vocab.get(ch, 0) for ch in smi[:max_sm_len]]
        if len(sm_idx) < max_sm_len:
            sm_idx.extend([0] * (max_sm_len - len(sm_idx)))
        d.sm = torch.tensor(sm_idx, dtype=torch.long).unsqueeze(0)
        d.w = torch.ones_like(d.y)
        d.raw_smiles = smi  # kept distinct from d.smiles (smiles_to_data's canonical form)
        data_list.append(d)
    return data_list
