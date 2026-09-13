"""Companion to crossmodal_model/benchmark/native_split.py: runs the thesis's own
graph+lang (no pretrain) model on the *exact same* per-seed splits MoLA's native data
organization produced (test/data/rederived_splits/<scaffold|random>/<dataset>/seed_<seed>/
{train,valid,test}.csv), instead of the fixed official split used by
thesis_model/benchmark/run_baselines.py / results/fase2_baselines.csv.

Same protocol as the "graph+lang" cell of run_baselines.py (hidden_dim=256,
num_layers=3, graph_backbone=gin, dropout=0.3, batch_size=32, lr=1e-3,
weight_decay=1e-4, warmup+plateau schedule, grad_clip=1.0, epochs<=100, patience=15,
standardize_target=True) -- only the split source changes.

Must be run AFTER crossmodal_model/benchmark/native_split.py has produced the CSV dumps
for the requested datasets/seeds/splitter.

Usage:
    python thesis_model/benchmark/native_split.py
    python thesis_model/benchmark/native_split.py --datasets freesolv --seeds 2025
"""
from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent  # test/thesis_model/benchmark/ -> test/
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import thesis_model.train.train as train_mod
from thesis_model.train.train import load_predefined_datasets, run_single

DATASETS = ["esol", "freesolv", "lipo"]
CSV_FIELDS = ["dataset", "config", "seed", "loss", "rmse", "nrmse", "mae", "mse", "r2", "elapsed_s"]
SPLIT_DUMP_ROOT = ROOT / "data" / "rederived_splits"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="graph+lang (no pretrain) on MoLA's native per-seed splits")
    parser.add_argument("--datasets", nargs="*", default=DATASETS, choices=DATASETS)
    parser.add_argument("--seeds", type=int, nargs="*", default=[2025, 2026, 2027])
    parser.add_argument(
        "--split-label", type=str, default="scaffold", choices=["scaffold", "random"],
        help="Selects the splitter subdir under data/rederived_splits/ to read -- this script does NOT "
        "itself re-derive a split. Make sure crossmodal_model/benchmark/native_split.py has been run "
        "with the matching --splitter before this, or you'll label the wrong data.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()
    args.config_name = f"graph+lang_native_{args.split_label}split"
    if args.out is None:
        args.out = str(ROOT / "results" / "mola" / f"mine_native_{args.split_label}split_benchmark.csv")
    if args.checkpoint_dir is None:
        args.checkpoint_dir = str(ROOT / "checkpoints" / f"native_{args.split_label}split_benchmark")
    return args


def _apply_shared_protocol(args: argparse.Namespace, cli: argparse.Namespace) -> None:
    # Identical to scripts/run_baselines.py's _apply_shared_protocol -- same protocol,
    # only the split source (--train-path/--val-path/--test-path) differs.
    args.epochs = cli.epochs
    args.patience = cli.patience
    args.device = cli.device
    args.lr_schedule = "plateau"
    args.warmup_epochs = 5
    args.plateau_factor = 0.5
    args.plateau_patience = 5
    args.min_lr_ratio = 0.01
    args.grad_clip = 1.0
    args.weight_decay = 1e-4
    args.standardize_target = True
    args.batch_size = 32
    args.deterministic = False


def _build_train_args(cli: argparse.Namespace) -> argparse.Namespace:
    args = train_mod.parse_args([])
    args.hidden_dim = 256
    args.num_layers = 3
    args.dropout = 0.3
    args.graph_backbone = "gin"
    args.language_model_name = "DeepChem/ChemBERTa-77M-MLM"
    args.use_graph, args.use_language = True, True
    _apply_shared_protocol(args, cli)
    return args


def _release_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _write_row(writer, handle, row) -> None:
    writer.writerow(row)
    handle.flush()


def _summary_row(dataset_name: str, config_name: str, per_seed: List[Dict[str, float]]) -> dict:
    row = {"dataset": dataset_name, "config": config_name, "seed": f"mean+/-std(n={len(per_seed)})"}
    for key in ("loss", "rmse", "nrmse", "mae", "mse", "r2"):
        vals = np.array([m[key] for m in per_seed if key in m], dtype="float64")
        row[key] = f"{vals.mean():.6f}+/-{vals.std():.6f}" if vals.size else ""
    row["elapsed_s"] = f"{sum(m.get('elapsed_s', 0.0) for m in per_seed):.1f}"
    return row


def main() -> None:
    cli = parse_args()
    out_path = Path(cli.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if cli.append and out_path.exists() else "w"

    with out_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if mode == "w":
            writer.writeheader()
            handle.flush()

        for dataset_name in cli.datasets:
            print(f"\n===== {dataset_name} =====", flush=True)
            per_seed: List[Dict[str, float]] = []
            for seed in cli.seeds:
                seed_dir = SPLIT_DUMP_ROOT / cli.split_label / dataset_name / f"seed_{seed}"
                train_p, val_p, test_p = seed_dir / "train.csv", seed_dir / "valid.csv", seed_dir / "test.csv"
                missing = [p for p in (train_p, val_p, test_p) if not p.exists()]
                if missing:
                    raise FileNotFoundError(
                        f"Missing MoLA-native split dump(s) for {dataset_name}/seed_{seed}: {missing}. "
                        "Run crossmodal_model/benchmark/native_split.py first for this dataset/seed/splitter."
                    )
                args = _build_train_args(cli)
                train_set, val_set, test_set = load_predefined_datasets(str(train_p), str(val_p), str(test_p))
                print(f"[{dataset_name} seed={seed}] loaded MoLA-native split: train={len(train_set)} val={len(val_set)} test={len(test_set)}")

                args.checkpoint_path = str(Path(cli.checkpoint_dir) / f"{dataset_name}_{cli.config_name}_s{seed}.pt")
                start = time.time()
                metrics = run_single(args, seed, train_set=train_set, val_set=val_set, test_set=test_set)
                metrics["elapsed_s"] = time.time() - start
                per_seed.append(metrics)
                _write_row(writer, handle, {"dataset": dataset_name, "config": cli.config_name, "seed": seed, **{k: metrics.get(k) for k in ("loss", "rmse", "nrmse", "mae", "mse", "r2", "elapsed_s")}})
                _release_gpu_memory()
            _write_row(writer, handle, _summary_row(dataset_name, cli.config_name, per_seed))

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
