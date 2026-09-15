"""Convert a CSV of SMILES to a list of PyG ``Data`` objects (OGB-style features).

The actual featurization lives in ``data_pipeline/features.py`` (shared with the
model-side encoders). This module keeps the CLI and re-exports the three helpers
every other module imports from here: ``smiles_to_data``, ``randomize_smiles``,
``canonicalize_smiles``.

Usage:
  python data_pipeline/convert_smiles_to_pyg.py --csv data/raw/lipo.csv --smiles-col smiles --target-col y --out data/lipo.graphs.pt
  python data_pipeline/convert_smiles_to_pyg.py --dataset-dir data/lipo --out data/lipo/graphs_from_smiles.pt
"""
from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path
from typing import Optional

import torch

from data_pipeline.features import (  # noqa: F401  (re-exported)
    canonicalize_smiles,
    randomize_smiles,
    smiles_to_data,
)


def find_csv_in_dataset_dir(dataset_dir: Path) -> Optional[Path]:
    raw = dataset_dir / "raw"
    if raw.exists():
        for f in sorted(raw.glob("*.csv")):
            return f
    for f in sorted(dataset_dir.glob("*.csv")):
        return f
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=str, default=None, help="Path to a CSV with a SMILES column")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Dataset folder to search for a CSV")
    parser.add_argument("--smiles-col", type=str, default="smiles")
    parser.add_argument("--target-col", type=str, default=None)
    parser.add_argument("--out", type=str, required=True, help="Output .pt file (a list of Data)")
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else (find_csv_in_dataset_dir(Path(args.dataset_dir)) if args.dataset_dir else None)
    if csv_path is None or not csv_path.exists():
        raise FileNotFoundError("CSV not found. Provide --csv or --dataset-dir pointing at a dataset CSV")

    open_f = gzip.open if str(csv_path).endswith(".gz") else open
    data_list = []
    with open_f(csv_path, "rt", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        smiles_col = args.smiles_col
        if smiles_col not in fields:
            smiles_col = next((c for c in fields if c and c.lower() in {"smiles", "smile", "canonical_smiles"}), fields[0] if fields else "smiles")
        target_col = args.target_col
        if target_col is None:
            target_col = next((c for c in fields if c and c != smiles_col), None)
        for row in reader:
            smi = (row.get(smiles_col) or "").strip()
            if not smi:
                continue
            target = row.get(target_col) if target_col else None
            data = smiles_to_data(smi, target)
            if data is not None:
                data_list.append(data)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data_list, outp)
    print(f"Saved {len(data_list)} graphs to {outp}")


if __name__ == "__main__":
    main()
