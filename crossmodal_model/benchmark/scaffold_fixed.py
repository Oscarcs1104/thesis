"""Decision experiment: MoLA (Graph + SMILES ONLY -- no fingerprint, no MolFormer, no
graph pretraining) vs. the thesis's own graph+lang (no pretrain) baseline, under
IDENTICAL scaffold splits, seeds, and training/eval protocol as results/fase2_baselines.csv.

Differences from crossmodal_model/data/featurize.py's own run_experiment(), and why:
  - Loads the *exact* official DeepChem scaffold-split CSVs already used by the rest of
    the thesis project (test/data/deepchem_molnet/<name>/csv/{train,valid,test}.csv)
    instead of re-deriving "a" scaffold split via dc.molnet.load_*(splitter_seed=cfg.seed)
    -- guarantees identical molecules per split, not just the same splitting *algorithm*.
  - Reuses common/repro.py's TargetStandardizer, warmup+plateau LR schedule, grad-norm
    clipping and regression_metrics (mse/rmse/mae/nrmse/r2) -- the exact same code
    fase2_baselines.csv was produced with, not a re-implementation of it.
  - Seeds torch/numpy/random per run (featurize.py's own --seed only ever fed the old
    random splitter, never the model init/shuffling), so seeds 2025/2026/2027 are a real
    multi-seed comparison, matching --seeds in thesis_model/benchmark/run_baselines.py.
  - Builds MoLA Graph + SMILES only -- the original MoLA fp+MolFormer (DNM) branch has
    been removed from crossmodal_model entirely, so there's no disabled-but-present
    third-modality branch sitting in the cross-layer attention.
  - hidden_dim=256, num_layers=3, batch_size=32, lr=1e-3, weight_decay=1e-4, epochs=100,
    patience=15 -- identical to the "graph+lang" cell of thesis_model/benchmark/run_baselines.py.

Known, unavoidable residual differences (report these alongside the numbers, don't
paper over them):
  - MoLA's own SMILES branch is a from-scratch char-level nn.Embedding + 1-layer
    TransformerEncoder; the thesis's language branch is a frozen pretrained ChemBERTa-77M.
    That is itself part of what "MoLA's architecture" means here -- not normalized away.
  - MoLA's graph branch consumes DeepChem's MolGraphConvFeaturizer node features (dense,
    GINConv, no edge_attr); the thesis's graph branch consumes its own categorical
    atom/bond features (GINEConv, edge-aware). Also part of the architecture comparison.
  - MolGraphConvFeaturizer cannot featurize single-heavy-atom molecules ("More than one
    atom should be present...") -- a handful of molecules (e.g. 1/902 in ESOL, ~3/511 in
    FreeSolv) are silently dropped from MoLA's split that the thesis's own pipeline keeps.
    Reported per-dataset below; it's a <1% effect, not expected to explain a large gap.

Usage:
    python crossmodal_model/benchmark/scaffold_fixed.py
    python crossmodal_model/benchmark/scaffold_fixed.py --datasets freesolv --seeds 2025
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
import torch
import torch.nn as nn

THIS_DIR = Path(__file__).resolve().parent                    # test/crossmodal_model/benchmark
TEST_ROOT = THIS_DIR.parent.parent                      # test/
if str(TEST_ROOT) not in sys.path:
    sys.path.append(str(TEST_ROOT))

from torch_geometric.loader import DataLoader as GeomDataLoader  # noqa: E402

from crossmodal_model.model.mola import MoLA  # noqa: E402
from crossmodal_model.train.core import DATASETS, build_datasets, evaluate, train_one_epoch  # noqa: E402
from common.repro import (  # noqa: E402
    TargetStandardizer,
    build_scheduler,
    seed_everything,
    step_scheduler,
)
from common.wandb_utils import add_wandb_args, wandb_finish, wandb_init, wandb_log  # noqa: E402

CSV_FIELDS = ["dataset", "config", "seed", "loss", "rmse", "nrmse", "mae", "mse", "r2", "elapsed_s"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MoLA Graph+SMILES-only vs. thesis graph+lang, same split/seeds/protocol")
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS.keys()), choices=list(DATASETS.keys()))
    parser.add_argument("--seeds", type=int, nargs="*", default=[2025, 2026, 2027])
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
    parser.add_argument(
        "--positional-smiles",
        action="store_true",
        help="A2: fix the SMILES branch (positional embedding + correct batch_first + padding-aware "
        "attention/pooling) instead of the original MoLA sm_transformer. Off by default so the "
        "original MoLA G+S numbers stay reproducible from this same script.",
    )
    parser.add_argument("--out", type=str, default=None, help="Default depends on --positional-smiles (keeps the A2 variant in a separate file)")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--append", action="store_true")
    add_wandb_args(parser)
    parser.set_defaults(wandb_project="mola-newtest-regression")
    args = parser.parse_args()
    # "NewTest" = this thesis's modified MoLA G+S. Suffix makes the A2 SMILES fix ablation
    # visible in filenames/checkpoints/wandb run names.
    args.config_name = "NewTest" + ("_a2fix" if args.positional_smiles else "")
    if args.out is None:
        args.out = str(TEST_ROOT / "results" / "mola" / f"{args.config_name}_benchmark.csv")
    if args.checkpoint_dir is None:
        args.checkpoint_dir = str(TEST_ROOT / "checkpoints" / "crossmodal" / args.config_name)
    return args


def run_one_seed(dataset_name, train_data, valid_data, test_data, vocab, seed: int, args) -> Dict[str, float]:
    seed_everything(seed, deterministic=False)
    device = args.device

    train_y = torch.stack([d.y.float().view(-1) for d in train_data])
    standardizer = TargetStandardizer(enabled=True).fit(train_y)
    target_range = (float(train_y.min()), float(train_y.max()))
    print(f"  [seed {seed}] train target stats: n={train_y.numel()} mean={train_y.mean():.3f} std={train_y.std():.3f} range={target_range}")

    model = MoLA(
        graph_dim=train_data[0].x.size(1),
        sm_vocab_size=len(vocab),  # crossmodal_model.data.featurize.build_vocab returns a flat {char: id} dict (+ "<pad>": 0)
        hidden_dim=args.hidden_dim,
        output_dim=1,
        num_layers=args.num_layers,
        positional_smiles=args.positional_smiles,
        max_sm_len=args.max_sm_len,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler("plateau", optimizer, args.warmup_epochs, args.epochs, args.min_lr_ratio, args.plateau_factor, args.plateau_patience)

    run_name = f"{dataset_name}-{args.config_name}-s{seed}"
    wandb_run = wandb_init(argparse.Namespace(**{**vars(args), "wandb_run_name": run_name}), config={**vars(args), "seed": seed, "dataset": dataset_name})

    train_loader = GeomDataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = GeomDataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
    test_loader = GeomDataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    checkpoint_path = Path(args.checkpoint_dir) / f"{dataset_name}_{args.config_name}_s{seed}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, standardizer, args.grad_clip)
        val_metrics = evaluate(model, valid_loader, criterion, device, standardizer, target_range)
        step_scheduler(scheduler, val_metrics["loss"])
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:03d} | lr={lr_now:.2e} | train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | val_rmse={val_metrics.get('rmse', float('nan')):.4f}")
        wandb_log(wandb_run, {"train/loss": train_loss, **{f"val/{k}": v for k, v in val_metrics.items()}, "lr": lr_now}, step=epoch)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
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
            train_data, valid_data, test_data, vocab = build_datasets(dataset_name, args.max_sm_len)
            per_seed: List[Dict[str, float]] = []
            for seed in args.seeds:
                start = time.time()
                metrics = run_one_seed(dataset_name, train_data, valid_data, test_data, vocab, seed, args)
                metrics["elapsed_s"] = time.time() - start
                per_seed.append(metrics)
                _write_row(writer, handle, {"dataset": dataset_name, "config": args.config_name, "seed": seed, **{k: metrics.get(k) for k in ("loss", "rmse", "nrmse", "mae", "mse", "r2", "elapsed_s")}})
            _write_row(writer, handle, _summary_row(dataset_name, args.config_name, per_seed))

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
