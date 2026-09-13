"""Second decision-experiment variant: same MoLA Graph+SMILES-only model/protocol as
scaffold_fixed.py, but using MoLA's OWN native data organization instead of the thesis's
fixed official-split CSVs -- i.e. DeepChem's ScaffoldSplitter (or RandomSplitter)
re-derives a fresh partition per seed (2025/2026/2027), instead of reusing test/'s one
fixed official split like scaffold_fixed.py / the rest of the thesis's --seeds protocol
does.

Also dumps each (dataset, seed) split to CSV under
test/data/rederived_splits/<scaffold|random>/<dataset>/seed_<seed>/{train,valid,test}.csv
so thesis_model/benchmark/native_split.py can point the thesis's own train.py at the
*exact same* molecules per split, for a same-data-organization comparison of both models.

IMPORTANT data-integrity note (found while building this): dc.molnet.load_freesolv/
load_delaney/load_lipo in this environment's deepchem==2.8.0 return y already z-scored,
and dc.trans.undo_transforms(y, transformers) does NOT reliably recover the true raw
units here -- verified directly (freesolv train y came back mean=0.141/std=0.853
instead of the real ~[-25, 3] kcal/mol range, both with reload=True *and* with a fully
wiped cache + reload=False, so it isn't just a stale-cache issue like the one
data_pipeline/download_deepchem_datasets.py's comments describe). So this script does
NOT call dc.molnet.load_*/undo_transforms at all: it reads the already-validated raw
SMILES+target pool from test/data/deepchem_molnet/<name>/csv/{train,valid,test}.csv
(known-good raw units, cross-checked multiple times this session), featurizes it once,
and calls dc.splits.ScaffoldSplitter directly with `seed=<seed>` -- this is still "the
data organization MoLA proposes" (DeepChem's own scaffold splitter, re-derived per seed)
without going through the loader path that's producing wrong-scale targets here.

featurize.py's own --seed only ever fed splitter_seed, never torch/numpy/random --
this script actually seeds each run (seed_everything) so "seed" controls model
init/shuffling too, not just which partition gets drawn.

Usage:
    python crossmodal_model/benchmark/native_split.py
    python crossmodal_model/benchmark/native_split.py --datasets freesolv --seeds 2025
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

THIS_DIR = Path(__file__).resolve().parent                    # test/crossmodal_model/benchmark
TEST_ROOT = THIS_DIR.parent.parent                      # test/
if str(TEST_ROOT) not in sys.path:
    sys.path.append(str(TEST_ROOT))

import deepchem as dc  # noqa: E402
from torch_geometric.loader import DataLoader as GeomDataLoader  # noqa: E402

from crossmodal_model.data.featurize import build_vocab, prepare_data  # noqa: E402
from crossmodal_model.model.mola import MoLA  # noqa: E402
from crossmodal_model.train.core import DATASETS, evaluate, train_one_epoch  # noqa: E402  -- reuse identical train/eval loop
from common.repro import (  # noqa: E402
    TargetStandardizer,
    build_scheduler,
    seed_everything,
    step_scheduler,
)
from common.wandb_utils import add_wandb_args, wandb_finish, wandb_init, wandb_log  # noqa: E402

CSV_FIELDS = ["dataset", "config", "seed", "loss", "rmse", "nrmse", "mae", "mse", "r2", "elapsed_s"]
SPLIT_DUMP_ROOT = TEST_ROOT / "data" / "rederived_splits"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NewTest on MoLA's own native (dc.splits, per-seed) data organization -- scaffold or random")
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS.keys()), choices=list(DATASETS.keys()))
    parser.add_argument("--seeds", type=int, nargs="*", default=[2025, 2026, 2027])
    parser.add_argument("--splitter", type=str, default="scaffold", choices=["scaffold", "random"], help="DeepChem splitter, re-derived fresh per seed")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--max-sm-len", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-patience", type=int, default=5)
    parser.add_argument("--min-lr-ratio", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--append", action="store_true")
    add_wandb_args(parser)
    parser.set_defaults(wandb_project="mola-newtest-regression")
    args = parser.parse_args()
    args.config_name = f"NewTest_{args.splitter}split"
    if args.out is None:
        args.out = str(TEST_ROOT / "results" / "mola" / f"{args.config_name}_benchmark.csv")
    if args.checkpoint_dir is None:
        args.checkpoint_dir = str(TEST_ROOT / "checkpoints" / "crossmodal" / args.config_name)
    return args


def dump_split_csv(ds, out_path: Path, target_col: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"smiles": list(ds.ids), target_col: np.asarray(ds.y).reshape(-1)})
    df.to_csv(out_path, index=False)


def _suspicious_stats(y) -> bool:
    """Same tripwire as data_pipeline/download_deepchem_datasets.py: a z-scored
    NormalizationTransformer output that failed to undo looks like mean~0, std~1 --
    every real ESOL/FreeSolv/Lipophilicity target we ship has |mean| or std well outside
    this band."""
    arr = np.asarray(y, dtype="float64").reshape(-1)
    mean, std = float(arr.mean()), float(arr.std())
    return abs(mean) < 0.5 and 0.5 <= std <= 2.0


def _load_raw_pool(dataset_name: str) -> "dc.data.NumpyDataset":
    """The full (train+valid+test recombined) SMILES+target pool, read straight from
    test/data/deepchem_molnet/<name>/csv/ -- already validated raw units this session
    (cross-checked against the values reported earlier in this conversation), featurized
    once here. Cached to disk (by dataset name only, not seed) since it's independent of
    which seed's scaffold split we're about to draw from it."""
    cfg = DATASETS[dataset_name]
    cache_path = TEST_ROOT / "data" / "featurized_pool" / f"{dataset_name}.pt"
    if cache_path.exists():
        obj = torch.load(cache_path, weights_only=False)
        return dc.data.NumpyDataset(X=obj["X"], y=obj["y"], ids=obj["ids"])

    csv_dir = TEST_ROOT / "data" / "deepchem_molnet" / cfg["dir"] / "csv"
    frames = [pd.read_csv(csv_dir / f"{split}.csv") for split in ("train", "valid", "test")]
    df = pd.concat(frames, ignore_index=True)
    smiles = df["smiles"].astype(str).tolist()
    y = df[cfg["target_col"]].astype(float).to_numpy().reshape(-1, 1)

    if _suspicious_stats(y):
        raise RuntimeError(
            f"[{dataset_name}] raw pool from {csv_dir} looks z-scored (mean~0, std~1) instead of "
            "raw units -- double-check the source CSVs before trusting anything downstream."
        )

    featurizer = dc.feat.MolGraphConvFeaturizer()
    X = featurizer.featurize(smiles)
    keep = [i for i, xi in enumerate(X) if hasattr(xi, "node_features") and hasattr(xi, "edge_index")]
    dropped = len(smiles) - len(keep)
    if dropped:
        print(f"  [{dataset_name}] dropped {dropped}/{len(smiles)} molecules MolGraphConvFeaturizer couldn't featurize")

    X_kept = np.array([X[i] for i in keep], dtype=object)
    y_kept = y[keep]
    ids_kept = np.array([smiles[i] for i in keep])

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"X": X_kept, "y": y_kept, "ids": ids_kept}, cache_path)
    return dc.data.NumpyDataset(X=X_kept, y=y_kept, ids=ids_kept)


_SPLITTERS = {"scaffold": dc.splits.ScaffoldSplitter, "random": dc.splits.RandomSplitter}


def load_and_dump(dataset_name: str, seed: int, splitter_name: str):
    cfg = DATASETS[dataset_name]
    pool = _load_raw_pool(dataset_name)
    splitter = _SPLITTERS[splitter_name]()
    train_ds, valid_ds, test_ds = splitter.train_valid_test_split(
        pool, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=seed
    )
    print(f"  [seed {seed}] {splitter_name.capitalize()}Splitter(seed={seed}) on the raw pool: train={len(train_ds.X)} valid={len(valid_ds.X)} test={len(test_ds.X)}")

    if _suspicious_stats(train_ds.y):
        raise RuntimeError(
            f"[{dataset_name} seed={seed}] train y looks z-scored (mean~0, std~1) instead of raw "
            "units -- aborting instead of silently reporting metrics on the wrong scale."
        )

    seed_dir = SPLIT_DUMP_ROOT / splitter_name / dataset_name / f"seed_{seed}"
    dump_split_csv(train_ds, seed_dir / "train.csv", cfg["target_col"])
    dump_split_csv(valid_ds, seed_dir / "valid.csv", cfg["target_col"])
    dump_split_csv(test_ds, seed_dir / "test.csv", cfg["target_col"])
    return train_ds, valid_ds, test_ds


def run_one(dataset_name: str, seed: int, args) -> Dict[str, float]:
    seed_everything(seed, deterministic=False)  # real seeding -- featurize.py's own --seed never did this
    device = args.device

    train_ds, valid_ds, test_ds = load_and_dump(dataset_name, seed, args.splitter)

    train_smiles, valid_smiles, test_smiles = list(train_ds.ids), list(valid_ds.ids), list(test_ds.ids)
    vocab = build_vocab(train_smiles + valid_smiles + test_smiles)

    def zero_fp(n):
        return np.zeros((n, 0), dtype=np.float32)  # Graph+SMILES only -- no third modality

    train_data = prepare_data(train_ds, zero_fp(len(train_ds.X)), train_smiles, vocab, max_sm_len=args.max_sm_len)
    valid_data = prepare_data(valid_ds, zero_fp(len(valid_ds.X)), valid_smiles, vocab, max_sm_len=args.max_sm_len)
    test_data = prepare_data(test_ds, zero_fp(len(test_ds.X)), test_smiles, vocab, max_sm_len=args.max_sm_len)

    train_y = torch.stack([d.y.float().view(-1) for d in train_data])
    standardizer = TargetStandardizer(enabled=True).fit(train_y)
    target_range = (float(train_y.min()), float(train_y.max()))
    print(f"  [seed {seed}] train target stats: n={train_y.numel()} mean={train_y.mean():.3f} std={train_y.std():.3f} range={target_range}")

    model = MoLA(
        graph_dim=train_data[0].x.size(1),
        sm_vocab_size=len(vocab),
        hidden_dim=args.hidden_dim,
        output_dim=1,
        num_layers=args.num_layers,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler("plateau", optimizer, args.warmup_epochs, args.epochs, args.min_lr_ratio, args.plateau_factor, args.plateau_patience)

    run_name = f"{dataset_name}-{args.config_name}-s{seed}"
    wandb_run = wandb_init(argparse.Namespace(**{**vars(args), "wandb_run_name": run_name}), config={**vars(args), "seed": seed, "dataset": dataset_name})

    train_loader = GeomDataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = GeomDataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
    test_loader = GeomDataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    best_val_loss, best_state, epochs_without_improvement = float("inf"), None, 0
    checkpoint_path = Path(args.checkpoint_dir) / f"{dataset_name}_{args.config_name}_s{seed}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, standardizer, args.grad_clip)
        val_metrics = evaluate(model, valid_loader, criterion, device, standardizer, target_range)
        step_scheduler(scheduler, val_metrics["loss"])
        print(f"  Epoch {epoch:03d} | lr={optimizer.param_groups[0]['lr']:.2e} | train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | val_rmse={val_metrics.get('rmse', float('nan')):.4f}")
        wandb_log(wandb_run, {"train/loss": train_loss, **{f"val/{k}": v for k, v in val_metrics.items()}, "lr": optimizer.param_groups[0]["lr"]}, step=epoch)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss, epochs_without_improvement = val_metrics["loss"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, checkpoint_path)
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            print(f"  [seed {seed}] early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, criterion, device, standardizer, target_range)
    print(
        f"  [seed {seed}] Test loss={test_metrics['loss']:.4f} | RMSE={test_metrics.get('rmse', float('nan')):.4f} "
        f"| NRMSE={test_metrics.get('nrmse', float('nan')):.4f} | MAE={test_metrics.get('mae', float('nan')):.4f} "
        f"| R2={test_metrics.get('r2', float('nan')):.4f}"
    )
    wandb_log(wandb_run, {f"test/{k}": v for k, v in test_metrics.items()})
    wandb_finish(wandb_run)
    return test_metrics


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
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append and out_path.exists() else "w"

    with out_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if mode == "w":
            writer.writeheader()
            handle.flush()

        for dataset_name in args.datasets:
            print(f"\n===== {dataset_name} =====", flush=True)
            per_seed: List[Dict[str, float]] = []
            for seed in args.seeds:
                start = time.time()
                metrics = run_one(dataset_name, seed, args)
                metrics["elapsed_s"] = time.time() - start
                per_seed.append(metrics)
                _write_row(writer, handle, {"dataset": dataset_name, "config": args.config_name, "seed": seed, **{k: metrics.get(k) for k in ("loss", "rmse", "nrmse", "mae", "mse", "r2", "elapsed_s")}})
            _write_row(writer, handle, _summary_row(dataset_name, args.config_name, per_seed))

    print(f"\nWrote {out_path}")
    print(f"Split CSV dumps under {SPLIT_DUMP_ROOT / args.splitter} (for thesis_model/benchmark/native_split.py to reuse on the same molecules)")


if __name__ == "__main__":
    main()
