"""Post-hoc scoring: adds a `predicted_property` column to the samples CSV that
generation/train.py already produced (results/mola/mola_generation_<dataset>_samples.csv),
using an INDEPENDENT predictor (checkpoints/crossmodal/mola_graph_smiles_a2_smiles_fix/
<dataset>_..._s2025.pt -- a separately-trained MoLA G+S regressor, never used for
generation).

Unlike eval_conditioning.py (which generates fresh candidates under +/-2std SHIFTED
conditions to test directional steering), this scores the candidates that were already
generated conditioned on each seed molecule's OWN true property -- i.e. it checks "does
the generated variant's predicted property match the property it was asked to
reproduce", not "does shifting the request move the property". Complementary check, same
independent-predictor principle.

Usage:
    python crossmodal_model/generation/score_property.py --dataset esol
    python crossmodal_model/generation/score_property.py --dataset esol freesolv lipo
"""
from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path
from typing import List

warnings.filterwarnings("ignore")

import numpy as np
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

THIS_DIR = Path(__file__).resolve().parent
TEST_ROOT = THIS_DIR.parent.parent
if str(TEST_ROOT) not in sys.path:
    sys.path.append(str(TEST_ROOT))

import deepchem as dc  # noqa: E402

from crossmodal_model.data.featurize import build_vocab as build_char_vocab  # noqa: E402
from crossmodal_model.generation.eval_conditioning import (  # noqa: E402 -- reuse the exact same predictor-loading/scoring code
    DEVICE, PREDICTOR_HIDDEN_DIM, PREDICTOR_MAX_SM_LEN, PREDICTOR_NUM_LAYERS,
    featurize_for_predictor, load_independent_predictor, predict_property,
)
from crossmodal_model.train.core import DATASETS, load_fixed_split  # noqa: E402


def score_dataset(dataset_name: str) -> None:
    samples_csv = TEST_ROOT / "results" / "mola" / f"mola_generation_{dataset_name}_samples.csv"
    predictor_ckpt = TEST_ROOT / "checkpoints" / "crossmodal" / "mola_graph_smiles_a2_smiles_fix" / f"{dataset_name}_mola_graph_smiles_a2_smiles_fix_s2025.pt"
    if not samples_csv.exists():
        print(f"[{dataset_name}] no samples CSV at {samples_csv}, skipping")
        return
    if not predictor_ckpt.exists():
        print(f"[{dataset_name}] no independent predictor checkpoint at {predictor_ckpt}, skipping")
        return

    cfg = DATASETS[dataset_name]
    csv_dir = TEST_ROOT / "data" / "deepchem_molnet" / cfg["dir"] / "csv"
    featurizer = dc.feat.MolGraphConvFeaturizer()
    train_ds = load_fixed_split(csv_dir / "train.csv", cfg["target_col"], featurizer)
    valid_ds = load_fixed_split(csv_dir / "valid.csv", cfg["target_col"], featurizer)
    test_ds = load_fixed_split(csv_dir / "test.csv", cfg["target_col"], featurizer)
    train_y_raw = np.asarray(train_ds.y, dtype="float64").reshape(-1)
    predictor_char_vocab = build_char_vocab(list(train_ds.ids) + list(valid_ds.ids) + list(test_ds.ids))

    import crossmodal_model.generation.eval_conditioning as ecs

    ecs.PREDICTOR_CKPT_PATH = predictor_ckpt  # point the reused loader at this dataset's own checkpoint
    predictor, standardizer = load_independent_predictor(train_y_raw, predictor_char_vocab)
    print(f"[{dataset_name}] independent predictor loaded cleanly (train mean={train_y_raw.mean():.3f} std={train_y_raw.std():.3f})")

    with samples_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        row["predicted_property"] = ""
        if row.get("valid", "").lower() == "true" and row.get("generated_canonical"):
            pred_data = featurize_for_predictor(row["generated_canonical"], featurizer, predictor_char_vocab, DEVICE)
            if pred_data is not None:
                row["predicted_property"] = predict_property(predictor, standardizer, pred_data)

    fieldnames = list(rows[0].keys())
    with samples_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[{dataset_name}] added predicted_property to {len(rows)} rows in {samples_csv}")

    scored = [r for r in rows if r["predicted_property"] != ""]
    target = np.array([float(r["target_property"]) for r in scored])
    predicted = np.array([float(r["predicted_property"]) for r in scored])
    errors = predicted - target
    pearson_r = float(np.corrcoef(target, predicted)[0, 1]) if len(scored) > 2 else float("nan")
    print(
        f"[{dataset_name}] {len(scored)}/{len(rows)} candidates scored | "
        f"target-vs-predicted Pearson r={pearson_r:.3f} | "
        f"mean|error|={np.abs(errors).mean():.3f} | mean error={errors.mean():+.3f} (bias)"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score generated candidates' property with an independent predictor")
    parser.add_argument("--dataset", nargs="+", default=list(DATASETS.keys()), choices=list(DATASETS.keys()))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for dataset_name in args.dataset:
        score_dataset(dataset_name)


if __name__ == "__main__":
    main()
