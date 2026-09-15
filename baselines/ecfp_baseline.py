"""Classical reference baselines: ECFP4 + RDKit descriptors -> XGBoost or MLP.

This is the point of comparison the whole thesis is judged against: if the
multimodal model can't beat a gradient-boosted tree on Morgan fingerprints
(minutes on CPU), the fusion isn't buying anything. Runs through the SAME
predefined scaffold split and the SAME metrics as every neural config in
scripts/run_baselines.py, so every row of the results table is comparable.
"""
from __future__ import annotations

import csv
from typing import List, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")

_DESC_NAMES = [name for name, _ in Descriptors._descList]
_DESC_CALC = MolecularDescriptorCalculator(_DESC_NAMES)
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)  # ECFP4


def _read_split_csv(path: str) -> Tuple[List[str], np.ndarray]:
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        smi_col = next((c for c in fields if c and c.lower() in {"smiles", "smile", "canonical_smiles"}), fields[0])
        tgt_col = next((c for c in fields if c and c != smi_col), None)
        smiles, targets = [], []
        for row in reader:
            s = (row.get(smi_col) or "").strip()
            try:
                y = float(row.get(tgt_col))
            except (TypeError, ValueError):
                continue
            if s:
                smiles.append(s)
                targets.append(y)
    return smiles, np.asarray(targets, dtype=np.float64)


def _features(smiles: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fps, descs, keep = [], [], []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        arr = np.zeros((2048,), dtype=np.float32)
        ConvertToNumpyArray(_MORGAN.GetFingerprint(mol), arr)
        d = np.asarray(_DESC_CALC.CalcDescriptors(mol), dtype=np.float64)
        d[~np.isfinite(d)] = 0.0
        np.clip(d, -1e8, 1e8, out=d)
        fps.append(arr)
        descs.append(d)
        keep.append(i)
    return np.vstack(fps), np.vstack(descs), np.asarray(keep)


def run_ecfp_baseline(train_csv: str, val_csv: str, test_csv: str, kind: str, seed: int, target_std: float) -> dict:
    from training.repro import regression_metrics  # local import: keeps rdkit-only callers light

    kind = kind.lower()
    tr_smi, tr_y = _read_split_csv(train_csv)
    va_smi, va_y = _read_split_csv(val_csv)
    te_smi, te_y = _read_split_csv(test_csv)

    (tr_fp, tr_d, tr_k), (va_fp, va_d, va_k), (te_fp, te_d, te_k) = _features(tr_smi), _features(va_smi), _features(te_smi)
    tr_y, va_y, te_y = tr_y[tr_k], va_y[va_k], te_y[te_k]

    imputer = SimpleImputer(strategy="median").fit(tr_d)
    scaler = StandardScaler().fit(imputer.transform(tr_d))

    def X(fp, d):
        return np.hstack([fp, scaler.transform(imputer.transform(d)).astype(np.float32)])

    Xtr, Xva, Xte = X(tr_fp, tr_d), X(va_fp, va_d), X(te_fp, te_d)

    if kind == "xgb":
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=2000, learning_rate=0.05, max_depth=6, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
            objective="reg:squarederror", eval_metric="rmse",
            early_stopping_rounds=100, tree_method="hist", n_jobs=-1, random_state=seed,
        )
        model.fit(Xtr, tr_y, eval_set=[(Xva, va_y)], verbose=False)
        pred = model.predict(Xte)
    elif kind == "mlp":
        model = MLPRegressor(
            hidden_layer_sizes=(512, 256), activation="relu", alpha=1e-4,
            batch_size=32, learning_rate_init=1e-3, max_iter=300,
            early_stopping=True, n_iter_no_change=20, random_state=seed,
        )
        model.fit(np.vstack([Xtr, Xva]), np.concatenate([tr_y, va_y]))
        pred = model.predict(Xte)
    else:
        raise ValueError(f"unknown ecfp baseline kind: {kind!r} (expected 'xgb' or 'mlp')")

    import torch

    m = regression_metrics(torch.tensor(pred), torch.tensor(te_y), target_std)
    m["loss"] = m["mse"]
    return m


def train_target_std(train_csv: str) -> float:
    _, y = _read_split_csv(train_csv)
    return float(np.std(y, ddof=1)) if y.size > 1 else 0.0
