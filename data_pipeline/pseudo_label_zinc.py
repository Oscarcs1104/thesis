"""Pseudo-label a SMILES CSV (e.g. ZINC-250k) with a trained predictor's estimate,
for fine-tuning the conditional generator (plan Day 10).

    python data_pipeline/pseudo_label_zinc.py \
        --predictor-checkpoint checkpoints/graph+lang_delaney_s2025.pt \
        --smiles-csv data/zinc15_250K.csv \
        --out data/zinc15_250K.pseudo_esol.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List

import torch
from torch_geometric.loader import DataLoader as GeomDataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data_pipeline.features import smiles_to_data
from model.model import build_model_from_args


class _NS:
    def __init__(self, d):
        self.__dict__.update(d)


def _load_predictor(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw = dict(ckpt.get("args", {}))
    for k, v in {
        "hidden_dim": 256, "output_dim": 1, "num_layers": 3, "dropout": 0.3,
        "graph_backbone": "gin", "language_backbone": "huggingface",
        "language_model_name": "DeepChem/ChemBERTa-77M-MLM",
        "freeze_language_backbone": True, "trust_remote_code": False,
        "use_graph": True, "use_language": True,
    }.items():
        raw.setdefault(k, v)
    model = build_model_from_args(_NS(raw)).to(device)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict"))
    model.load_state_dict(state, strict=False)
    model.eval()

    std = ckpt.get("target_standardizer")
    mean = float(std["mean"].view(-1)[0]) if std and std.get("mean") is not None else 0.0
    scale = float(std["std"].view(-1)[0]) if std and std.get("std") is not None else 1.0
    return model, mean, scale


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictor-checkpoint", required=True)
    ap.add_argument("--smiles-csv", default="data/zinc15_250K.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--value-name", default="y_pred")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, mean, scale = _load_predictor(args.predictor_checkpoint, args.device)

    with open(args.smiles_csv, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        col = next((c for c in fields if c and c.lower() in {"smiles", "smile", "canonical_smiles"}), fields[0])
        rows = [(r.get(col) or "").strip() for r in reader]
    rows = [s for s in rows if s]
    if args.limit:
        rows = rows[: args.limit]

    graphs, kept_smiles = [], []
    for s in rows:
        g = smiles_to_data(s)
        if g is None:
            continue
        graphs.append(g)
        kept_smiles.append(g.smiles)
    print(f"{len(graphs)}/{len(rows)} SMILES featurized")

    preds: List[float] = []
    loader = GeomDataLoader(graphs, batch_size=args.batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(args.device)
            out = model(batch).view(-1).cpu()
            preds.extend((out * scale + mean).tolist())

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["smiles", args.value_name])
        for s, y in zip(kept_smiles, preds):
            w.writerow([s, f"{y:.6f}"])
    print(f"Wrote {len(preds)} pseudo-labels to {outp}")


if __name__ == "__main__":
    main()
