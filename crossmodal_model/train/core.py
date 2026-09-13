"""Shared regression data-loading + train/eval loop for crossmodal_model's benchmarks.
Split out of what used to be run_scaffold_benchmark.py so both
crossmodal_model/benchmark/scaffold_fixed.py (fixed official split) and
crossmodal_model/benchmark/native_split.py (re-derived per-seed split) reuse the exact
same featurization/training code instead of importing it from one another's entrypoint.

Mirrors thesis_model/train/train.py's protocol exactly (same common/repro.py helpers).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

import deepchem as dc

from crossmodal_model.data.featurize import build_vocab, prepare_data

TEST_ROOT = Path(__file__).resolve().parent.parent.parent  # test/crossmodal_model/train/ -> test/

DATASETS: Dict[str, Dict[str, str]] = {
    "esol": {"dir": "delaney", "target_col": "measured log solubility in mols per litre"},
    "freesolv": {"dir": "freesolv", "target_col": "y"},
    "lipo": {"dir": "lipo", "target_col": "exp"},
}


# --------------------------------------------------------------------------- #
# Data: featurize the thesis's own official-split CSVs (not a re-derived split)
# --------------------------------------------------------------------------- #
def load_fixed_split(csv_path: Path, target_col: str, featurizer) -> "dc.data.NumpyDataset":
    df = pd.read_csv(csv_path)
    smiles = df["smiles"].astype(str).tolist()
    y = df[target_col].astype(float).to_numpy().reshape(-1, 1)

    X = featurizer.featurize(smiles)
    keep = [i for i, xi in enumerate(X) if hasattr(xi, "node_features") and hasattr(xi, "edge_index")]
    dropped = len(smiles) - len(keep)
    if dropped:
        print(f"  [{csv_path.name}] dropped {dropped}/{len(smiles)} molecules MolGraphConvFeaturizer couldn't featurize")

    X_kept = np.array([X[i] for i in keep], dtype=object)
    y_kept = y[keep]
    ids_kept = np.array([smiles[i] for i in keep])
    w_kept = np.ones_like(y_kept)
    return dc.data.NumpyDataset(X=X_kept, y=y_kept, w=w_kept, ids=ids_kept)


def build_datasets(dataset_name: str, max_sm_len: int):
    cfg = DATASETS[dataset_name]
    csv_dir = TEST_ROOT / "data" / "deepchem_molnet" / cfg["dir"] / "csv"
    featurizer = dc.feat.MolGraphConvFeaturizer()

    print(f"Featurizing {dataset_name} from {csv_dir} (official scaffold split, same CSVs as fase2_baselines.csv)...")
    train_ds = load_fixed_split(csv_dir / "train.csv", cfg["target_col"], featurizer)
    valid_ds = load_fixed_split(csv_dir / "valid.csv", cfg["target_col"], featurizer)
    test_ds = load_fixed_split(csv_dir / "test.csv", cfg["target_col"], featurizer)
    print(f"  sizes: train={len(train_ds.X)} valid={len(valid_ds.X)} test={len(test_ds.X)}")

    train_smiles = list(train_ds.ids)
    valid_smiles = list(valid_ds.ids)
    test_smiles = list(test_ds.ids)
    vocab = build_vocab(train_smiles + valid_smiles + test_smiles)

    def zero_fp(n):
        return np.zeros((n, 0), dtype=np.float32)  # Graph+SMILES only -- no third modality

    train_data = prepare_data(train_ds, zero_fp(len(train_ds.X)), train_smiles, vocab, max_sm_len=max_sm_len)
    valid_data = prepare_data(valid_ds, zero_fp(len(valid_ds.X)), valid_smiles, vocab, max_sm_len=max_sm_len)
    test_data = prepare_data(test_ds, zero_fp(len(test_ds.X)), test_smiles, vocab, max_sm_len=max_sm_len)
    return train_data, valid_data, test_data, vocab


# --------------------------------------------------------------------------- #
# Train / eval -- mirrors thesis_model/train/train.py's protocol exactly (same helpers)
# --------------------------------------------------------------------------- #
def _forward_preds(model, batch) -> torch.Tensor:
    return model(batch)[-1]  # MoLA.forward returns [fused_feat, layer_weights, attn_map, fused_out_final]


def train_one_epoch(model, loader, optimizer, criterion, device, standardizer, grad_clip) -> float:
    model.train()
    total_loss, total_items = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        preds = _forward_preds(model, batch)
        targets_orig = batch.y.float().view_as(preds)
        targets_std = standardizer.transform(targets_orig)
        loss = criterion(preds, targets_std)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        total_items += batch.num_graphs
    return total_loss / max(total_items, 1)


def evaluate(model, loader, criterion, device, standardizer, target_range) -> Dict[str, float]:
    from common.repro import regression_metrics

    model.eval()
    total_loss, total_items = 0.0, 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds = _forward_preds(model, batch)
            targets_orig = batch.y.float().view_as(preds)
            targets_std = standardizer.transform(targets_orig)
            loss = criterion(preds, targets_std)
            preds_orig = standardizer.inverse_transform(preds)
            all_preds.append(preds_orig.detach().cpu())
            all_targets.append(targets_orig.detach().cpu())
            total_loss += loss.item() * batch.num_graphs
            total_items += batch.num_graphs
    metrics = {"loss": total_loss / max(total_items, 1)}
    metrics.update(regression_metrics(torch.cat(all_preds), torch.cat(all_targets), target_range))
    return metrics
