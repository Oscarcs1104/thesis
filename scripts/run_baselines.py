"""FASE 2: standardized baseline matrix.

Runs the SAME seeds / patience / LR schedule / split policy across every
architecture ablation and dataset, and logs everything to one CSV -- so every
row is a fair comparison instead of hand-run, hard-to-compare one-offs.

Configs (per dataset):
  graph-only            GNN alone, no language branch
  lang-only-frozen      ChemBERTa alone, frozen (main model's default language setup)
  lang-only-unfrozen    ChemBERTa alone, fine-tuned end to end (D2), lr=1e-5
  graph+lang            both branches, concat fusion, no pretrain -- the reference baseline
  graph+lang+pretrain   both branches + ZINC-pretrained graph encoder, linear-probed
                        for --linear-probe-epochs then unfrozen at a reduced lr

precomputed-molformer is intentionally NOT included: it needs
data_pipeline/precompute_molformer_embeddings.py run first (downloads and runs
a 1.1B-parameter model, its own multi-minute-to-hour step). Run that ablation
separately via training/train_precomputed_molformer.py once you've generated
data/deepchem_molnet/<dataset>/csv/{train,valid,test}.molformer_emb.pt.

Usage:
    python scripts/run_baselines.py
    python scripts/run_baselines.py --datasets esol lipo --configs graph-only graph+lang --seeds 2025
    python scripts/run_baselines.py --out results/fase2_baselines.csv
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

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import training.train as train_mod
from training.train import _target_range, load_predefined_datasets, resolve_predefined_split, run_single

DATASETS: Dict[str, str] = {
    "esol": "data/deepchem_molnet/delaney",
    "freesolv": "data/deepchem_molnet/freesolv",
    "lipo": "data/deepchem_molnet/lipo",
}
ALL_CONFIGS = ["graph-only", "lang-only-frozen", "lang-only-unfrozen", "graph+lang", "graph+lang+pretrain"]
CSV_FIELDS = ["dataset", "config", "seed", "loss", "rmse", "nrmse", "mae", "mse", "elapsed_s"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FASE 2 standardized baseline matrix")
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS.keys()), choices=list(DATASETS.keys()))
    parser.add_argument("--configs", nargs="*", default=ALL_CONFIGS, choices=ALL_CONFIGS)
    parser.add_argument("--seeds", type=int, nargs="*", default=[2025, 2026, 2027])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--graph-pretrained-checkpoint", type=str, default="checkpoints/multitask_graph_pretrain.pt")
    parser.add_argument("--linear-probe-epochs", type=int, default=5)
    parser.add_argument("--unfreeze-lr-mult", type=float, default=0.1)
    parser.add_argument("--out", type=str, default="results/fase2_baselines.csv")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/baselines")
    parser.add_argument("--append", action="store_true", help="Append to --out instead of overwriting it")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Shared protocol -- identical across every dataset x config (D1/D4 standardization)
# --------------------------------------------------------------------------- #
def _apply_shared_protocol(args: argparse.Namespace, cli: argparse.Namespace) -> None:
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
    args.split = "scaffold"  # fallback only -- every dataset here has a predefined split


def _build_train_args(dataset_dir: str, config: str, cli: argparse.Namespace) -> argparse.Namespace:
    args = train_mod.parse_args([])  # pure CLI defaults, no sys.argv involved
    args.dataset_dir = dataset_dir
    args.hidden_dim = 256
    args.num_layers = 3
    args.dropout = 0.3
    args.graph_backbone = "gin"
    args.language_model_name = "DeepChem/ChemBERTa-77M-MLM"
    _apply_shared_protocol(args, cli)

    if config == "graph-only":
        args.use_graph, args.use_language, args.language_backbone = True, False, "none"
    elif config == "lang-only-frozen":
        args.use_graph, args.use_language, args.freeze_language_backbone = False, True, True
    elif config == "lang-only-unfrozen":
        args.use_graph, args.use_language, args.freeze_language_backbone = False, True, False
        args.lr = 1e-5  # D2: full fine-tune needs a much smaller lr than a frozen-backbone head
    elif config == "graph+lang":
        args.use_graph, args.use_language = True, True
    elif config == "graph+lang+pretrain":
        args.use_graph, args.use_language = True, True
        if not Path(cli.graph_pretrained_checkpoint).exists():
            raise FileNotFoundError(f"--graph-pretrained-checkpoint not found: {cli.graph_pretrained_checkpoint}")
        args.graph_pretrained_checkpoint = cli.graph_pretrained_checkpoint
        args.linear_probe_epochs = cli.linear_probe_epochs
        args.unfreeze_lr_mult = cli.unfreeze_lr_mult
    else:
        raise ValueError(f"Unhandled config for _build_train_args: {config}")
    return args


def _write_row(writer: "csv.DictWriter", handle, row: dict) -> None:
    writer.writerow(row)
    handle.flush()


def _release_gpu_memory() -> None:
    """Belt-and-suspenders between the many models this script builds in one process
    (several re-loading ChemBERTa-77M): each run_single() call should already drop
    its own model/optimizer once it returns, but a lingering autograd-graph reference
    cycle can delay that past a plain refcount drop, and PyTorch's CUDA caching
    allocator won't hand freed-but-cached blocks back to the driver on its own.
    Forcing both here keeps VRAM flat across the whole matrix instead of accumulating
    toward an OOM many cells in."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_config(dataset_name: str, dataset_dir: str, config: str, cli: argparse.Namespace, writer, handle) -> List[Dict[str, float]]:
    per_seed: List[Dict[str, float]] = []

    args = _build_train_args(dataset_dir, config, cli)
    predefined = resolve_predefined_split(args)
    train_set, val_set, test_set = load_predefined_datasets(*predefined)
    for seed in cli.seeds:
        args.checkpoint_path = str(Path(cli.checkpoint_dir) / f"{dataset_name}_{config}_s{seed}.pt")
        start = time.time()
        metrics = run_single(args, seed, train_set=train_set, val_set=val_set, test_set=test_set)
        metrics["elapsed_s"] = time.time() - start
        per_seed.append(metrics)
        _write_row(writer, handle, {"dataset": dataset_name, "config": config, "seed": seed, **{k: metrics.get(k) for k in ("loss", "rmse", "nrmse", "mae", "mse", "elapsed_s")}})
        _release_gpu_memory()
    return per_seed


def _summary_row(dataset_name: str, config: str, per_seed: List[Dict[str, float]]) -> dict:
    row = {"dataset": dataset_name, "config": config, "seed": f"mean+/-std(n={len(per_seed)})"}
    for key in ("loss", "rmse", "nrmse", "mae", "mse"):
        vals = np.array([m[key] for m in per_seed if key in m], dtype="float64")
        # ASCII "+/-", not the unicode "±": Windows consoles/files default to a non-UTF-8
        # codepage often enough that "±" round-trips as mangled bytes (seen firsthand here).
        row[key] = f"{vals.mean():.6f}+/-{vals.std():.6f}" if vals.size else ""
    row["elapsed_s"] = f"{sum(m.get('elapsed_s', 0.0) for m in per_seed):.1f}"
    return row


def main() -> None:
    cli = parse_args()
    Path(cli.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(cli.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if cli.append and out_path.exists() else "w"
    with out_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if mode == "w":
            writer.writeheader()
            handle.flush()

        all_results: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
        total = len(cli.datasets) * len(cli.configs)
        done = 0
        overall_start = time.time()
        for dataset_name in cli.datasets:
            dataset_dir = DATASETS[dataset_name]
            all_results[dataset_name] = {}
            for config in cli.configs:
                done += 1
                print(f"\n===== [{done}/{total}] {dataset_name} / {config} =====", flush=True)
                run_start = time.time()
                try:
                    per_seed = _run_config(dataset_name, dataset_dir, config, cli, writer, handle)
                except Exception as exc:  # keep the matrix going even if one cell fails
                    print(f"  FAILED: {dataset_name}/{config}: {exc!r}", flush=True)
                    _write_row(writer, handle, {"dataset": dataset_name, "config": config, "seed": "ERROR", "loss": str(exc)[:200]})
                    continue
                all_results[dataset_name][config] = per_seed
                _write_row(writer, handle, _summary_row(dataset_name, config, per_seed))
                print(f"  done in {time.time() - run_start:.1f}s (total elapsed {time.time() - overall_start:.1f}s)", flush=True)

    print(f"\nWrote {out_path}")
    print("\n=== RMSE mean (test), by dataset x config ===")
    header = f"{'config':<22s}" + "".join(f"{d:>14s}" for d in cli.datasets)
    print(header)
    for config in cli.configs:
        cells = []
        for dataset_name in cli.datasets:
            per_seed = all_results.get(dataset_name, {}).get(config)
            if not per_seed:
                cells.append(f"{'--':>14s}")
                continue
            rmses = np.array([m["rmse"] for m in per_seed], dtype="float64")
            cells.append(f"{rmses.mean():>8.4f}+/-{rmses.std():<5.3f}")
        print(f"{config:<22s}" + "".join(cells))


if __name__ == "__main__":
    main()
