"""HybridMoLA-PretrainedSMILES benchmark on RE-DERIVED splits (scaffold or random) --
reads the split dumps from native_split.py, same protocol as native_split_hybrid.py, but
with the SMILES branch swapped for a pretrained transformer (ChemBERTa, frozen by
default) -- see crossmodal_model/model/mola_hybrid_pretrained_smiles.py.

Usage:
    python crossmodal_model/benchmark/native_split_hybrid_pretrained_smiles.py --splitter random
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

THIS_DIR = Path(__file__).resolve().parent
TEST_ROOT = THIS_DIR.parent.parent
if str(TEST_ROOT) not in sys.path:
    sys.path.append(str(TEST_ROOT))

from torch_geometric.loader import DataLoader as GeomDataLoader  # noqa: E402

from data_pipeline.convert_smiles_to_pyg import smiles_to_data  # noqa: E402
from crossmodal_model.model.mola_hybrid_pretrained_smiles import HybridMoLAPretrainedSMILES  # noqa: E402
from crossmodal_model.train.core import DATASETS  # noqa: E402
from common.repro import TargetStandardizer, build_scheduler, regression_metrics, seed_everything, step_scheduler  # noqa: E402
from common.wandb_utils import add_wandb_args, wandb_finish, wandb_init, wandb_log  # noqa: E402

SPLIT_DUMP_ROOT = TEST_ROOT / "data" / "rederived_splits"
CSV_FIELDS = ["dataset", "config", "seed", "loss", "rmse", "nrmse", "mae", "mse", "r2", "elapsed_s"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HybridMoLA (pretrained SMILES branch) on re-derived splits")
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS.keys()), choices=list(DATASETS.keys()))
    parser.add_argument("--seeds", type=int, nargs="*", default=[2025, 2026, 2027])
    parser.add_argument("--splitter", type=str, default="scaffold", choices=["scaffold", "random"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-patience", type=int, default=5)
    parser.add_argument("--min-lr-ratio", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--graph-backbone", type=str, default="gin", choices=["gcn", "gat", "gatv2", "gin"])
    parser.add_argument("--graph-pooling", type=str, default="add", choices=["mean", "add", "sum", "max", "mean_max"])
    parser.add_argument("--smiles-model-name", type=str, default="DeepChem/ChemBERTa-77M-MLM")
    parser.add_argument("--unfreeze-smiles-backbone", dest="freeze_smiles_backbone", action="store_false")
    parser.add_argument("--smiles-trust-remote-code", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--append", action="store_true")
    add_wandb_args(parser)
    parser.set_defaults(wandb_project="mola-newtest-regression", freeze_smiles_backbone=True)
    args = parser.parse_args()
    args.config_name = f"NewTestHybridPretrainedSMILES_{args.splitter}split"
    if args.out is None:
        args.out = str(TEST_ROOT / "results" / "mola" / f"{args.config_name}_benchmark.csv")
    if args.checkpoint_dir is None:
        args.checkpoint_dir = str(TEST_ROOT / "checkpoints" / "crossmodal" / args.config_name)
    return args


def load_dump(dataset_name: str, seed: int, splitter: str, cfg: Dict[str, str]):
    seed_dir = SPLIT_DUMP_ROOT / splitter / dataset_name / f"seed_{seed}"
    train_p, val_p, test_p = seed_dir / "train.csv", seed_dir / "valid.csv", seed_dir / "test.csv"
    missing = [p for p in (train_p, val_p, test_p) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing split dump(s) for {dataset_name}/seed_{seed}: {missing}. "
            f"Run crossmodal_model/benchmark/native_split.py --splitter {splitter} first."
        )

    def _load(p):
        df = pd.read_csv(p)
        smiles = df["smiles"].astype(str).tolist()
        y = df[cfg["target_col"]].astype(float).tolist()
        data_list = []
        for smi, target in zip(smiles, y):
            d = smiles_to_data(smi, target=float(target))
            if d is not None:
                data_list.append(d)
        return data_list

    return _load(train_p), _load(val_p), _load(test_p)


def _forward_preds(model, batch) -> torch.Tensor:
    return model(batch)[-1]


def train_one_epoch(model, loader, optimizer, criterion, device, standardizer, grad_clip) -> float:
    model.train()
    total_loss, total_items = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        preds = _forward_preds(model, batch)
        targets_std = standardizer.transform(batch.y.float().view_as(preds))
        loss = criterion(preds, targets_std)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        total_items += batch.num_graphs
    return total_loss / max(total_items, 1)


def evaluate(model, loader, criterion, device, standardizer, target_range) -> Dict[str, float]:
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
            all_preds.append(standardizer.inverse_transform(preds).detach().cpu())
            all_targets.append(targets_orig.detach().cpu())
            total_loss += loss.item() * batch.num_graphs
            total_items += batch.num_graphs
    metrics = {"loss": total_loss / max(total_items, 1)}
    metrics.update(regression_metrics(torch.cat(all_preds), torch.cat(all_targets), target_range))
    return metrics


def run_one(dataset_name: str, seed: int, args) -> Dict[str, float]:
    seed_everything(seed, deterministic=False)
    device = args.device
    cfg = DATASETS[dataset_name]
    train_data, valid_data, test_data = load_dump(dataset_name, seed, args.splitter, cfg)
    print(f"  [seed {seed}] sizes: train={len(train_data)} valid={len(valid_data)} test={len(test_data)}")

    train_y_t = torch.stack([d.y.float().view(-1) for d in train_data])
    standardizer = TargetStandardizer(enabled=True).fit(train_y_t)
    target_range = (float(train_y_t.min()), float(train_y_t.max()))
    print(f"  [seed {seed}] train target stats: n={train_y_t.numel()} mean={train_y_t.mean():.3f} std={train_y_t.std():.3f} range={target_range}")

    model = HybridMoLAPretrainedSMILES(
        hidden_dim=args.hidden_dim, output_dim=1, num_layers=args.num_layers,
        graph_backbone=args.graph_backbone, graph_pooling=args.graph_pooling,
        smiles_model_name=args.smiles_model_name, freeze_smiles_backbone=args.freeze_smiles_backbone,
        smiles_trust_remote_code=args.smiles_trust_remote_code,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
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
            per_seed: List[Dict[str, float]] = []
            for seed in args.seeds:
                start = time.time()
                metrics = run_one(dataset_name, seed, args)
                metrics["elapsed_s"] = time.time() - start
                per_seed.append(metrics)
                _write_row(writer, handle, {"dataset": dataset_name, "config": args.config_name, "seed": seed, **{k: metrics.get(k) for k in ("loss", "rmse", "nrmse", "mae", "mse", "r2", "elapsed_s")}})
            _write_row(writer, handle, _summary_row(dataset_name, args.config_name, per_seed))

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
