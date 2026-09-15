"""Property-prediction ablation over PRETRAINED backbones, MoLA fusion, MoleculeNet.

Five configurations x three datasets x N seeds -> one CSV:

    chemberta        ChemBERTa-77M alone
    molformer        MoLFormer-XL alone
    gin              pretrained GIN alone (the reference the fusion has to beat)
    chemberta+gin    both, fused with MoLA cross-layer attention
    molformer+gin    both, fused with MoLA cross-layer attention

Graphs use the Hu et al. schema (data_pipeline/features_pretrain_gnn.py), because that
is what the GIN checkpoints were trained on -- not the OGB 9+3 schema the rest of the
repo uses.

Both backbones are FROZEN by default. Only the per-branch projection, the MoLA
cross-attention, the layer weights and the head are trained. That makes the table a
comparison of the pretrained REPRESENTATIONS rather than of how well each architecture
fine-tunes on a few hundred molecules -- and on FreeSolv-sized data, fine-tuning a 77M
transformer mostly measures how fast it overfits. --no-freeze-lm / --no-freeze-gin
switch it, but whatever is chosen has to hold across every row or the comparison stops
being about the backbones.

    python crossmodal_model/benchmark/pretrained_ablation.py
    python crossmodal_model/benchmark/pretrained_ablation.py --datasets esol --configs gin chemberta+gin
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

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from rdkit import RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

from torch_geometric.loader import DataLoader as GeomDataLoader  # noqa: E402

from common.repro import (  # noqa: E402
    TargetStandardizer,
    build_scheduler,
    regression_metrics,
    seed_everything,
    step_scheduler,
)
from crossmodal_model.model.mola_pretrained import CONFIGS, build_config  # noqa: E402
from crossmodal_model.train.core import DATASETS  # noqa: E402
from data_pipeline.features_pretrain_gnn import smiles_to_data_pretrain  # noqa: E402

CSV_FIELDS = ["dataset", "config", "seed", "rmse", "mae", "nrmse", "r2",
              "best_epoch", "n_params", "n_trainable", "elapsed_s"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="*", default=["esol", "freesolv", "lipo"], choices=list(DATASETS))
    p.add_argument("--configs", nargs="*", default=list(CONFIGS), choices=list(CONFIGS))
    p.add_argument("--seeds", nargs="*", type=int, default=[2025, 2026, 2027])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--gin-variant", type=str, default="contextpred")
    p.add_argument("--freeze-lm", dest="freeze_lm", action="store_true", default=True)
    p.add_argument("--no-freeze-lm", dest="freeze_lm", action="store_false")
    p.add_argument("--freeze-gin", dest="freeze_gin", action="store_true", default=True)
    p.add_argument("--no-freeze-gin", dest="freeze_gin", action="store_false")
    p.add_argument("--out", type=str, default=str(ROOT / "results" / "pretrained_ablation.csv"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_split(dataset: str) -> Dict[str, list]:
    cfg = DATASETS[dataset]
    csv_dir = ROOT / "data" / "deepchem_molnet" / cfg["dir"] / "csv"
    out = {}
    for split in ("train", "valid", "test"):
        df = pd.read_csv(csv_dir / f"{split}.csv")
        data = [smiles_to_data_pretrain(s, target=y)
                for s, y in zip(df["smiles"].astype(str), df[cfg["target_col"]].astype(float))]
        out[split] = [d for d in data if d is not None]
        dropped = len(df) - len(out[split])
        if dropped:
            print(f"    [{dataset}/{split}] dropped {dropped} unparseable molecules")
    return out


def run_one(dataset: str, config: str, seed: int, args) -> Dict[str, float]:
    seed_everything(seed, deterministic=False)
    device = args.device
    start = time.time()

    splits = load_split(dataset)
    train_y = torch.tensor([float(d.y) for d in splits["train"]])
    standardizer = TargetStandardizer(enabled=True).fit(train_y)
    target_std = float(train_y.std())

    model = build_config(
        config, hidden_dim=args.hidden_dim, output_dim=1, num_layers=args.num_layers,
        lm_freeze=args.freeze_lm, gin_variant=args.gin_variant, gin_freeze=args.freeze_gin,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    loaders = {
        k: GeomDataLoader(
            v, batch_size=args.batch_size, shuffle=(k == "train"),
            # BatchNorm1d cannot compute statistics from a single row, and a trailing
            # batch of one is common on FreeSolv-sized splits.
            drop_last=(k == "train" and len(v) > args.batch_size),
        )
        for k, v in splits.items()
    }

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = build_scheduler("plateau", optimizer, args.warmup_epochs, args.epochs, 0.01, 0.5, 5)

    def epoch(loader, train: bool) -> Dict[str, float]:
        model.train(train)
        preds_all, targets_all, total, n = [], [], 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            with torch.set_grad_enabled(train):
                preds = model(batch)[-1]
                targets = batch.y.float().view_as(preds)
                loss = criterion(preds, standardizer.transform(targets))
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            preds_all.append(standardizer.inverse_transform(preds).detach().cpu())
            targets_all.append(targets.detach().cpu())
            total += loss.item() * batch.num_graphs
            n += batch.num_graphs
        m = regression_metrics(torch.cat(preds_all), torch.cat(targets_all), target_std)
        m["loss"] = total / max(n, 1)
        return m

    best_rmse, best_epoch, best_state, stale = float("inf"), 0, None, 0
    for ep in range(1, args.epochs + 1):
        epoch(loaders["train"], True)
        val = epoch(loaders["valid"], False)
        step_scheduler(scheduler, val["rmse"])
        if val["rmse"] < best_rmse:
            best_rmse, best_epoch, stale = val["rmse"], ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test = epoch(loaders["test"], False)
    return {
        "dataset": dataset, "config": config, "seed": seed,
        "rmse": test["rmse"], "mae": test["mae"], "nrmse": test["nrmse"], "r2": test["r2"],
        "best_epoch": best_epoch, "n_params": n_params, "n_trainable": n_trainable,
        "elapsed_s": time.time() - start,
    }


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()

    rows: List[Dict[str, float]] = []
    for dataset in args.datasets:
        for config in args.configs:
            for seed in args.seeds:
                print(f"\n=== {dataset} | {config} | seed {seed} ===", flush=True)
                try:
                    row = run_one(dataset, config, seed, args)
                except Exception as exc:  # one dead cell must not lose the rest of the grid
                    print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
                    continue
                rows.append(row)
                print(f"  rmse={row['rmse']:.4f} mae={row['mae']:.4f} r2={row['r2']:.4f} "
                      f"({row['elapsed_s']:.0f}s, best epoch {row['best_epoch']}, "
                      f"{row['n_trainable']:,}/{row['n_params']:,} trainable)", flush=True)
                with out_path.open("a", newline="", encoding="utf-8") as fh:
                    w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
                    if write_header:
                        w.writeheader()
                        write_header = False
                    w.writerow(row)

    if not rows:
        print("\nNo runs completed.")
        return

    df = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print("RMSE, mean +/- std over seeds (lower is better)")
    print("=" * 78)
    pivot = df.groupby(["config", "dataset"])["rmse"].agg(["mean", "std", "count"])
    header = f"{'config':<16}" + "".join(f"{d:>20}" for d in args.datasets)
    print(header)
    print("-" * len(header))
    for config in args.configs:
        line = f"{config:<16}"
        for dataset in args.datasets:
            if (config, dataset) in pivot.index:
                r = pivot.loc[(config, dataset)]
                sd = 0.0 if pd.isna(r["std"]) else r["std"]
                line += f"{r['mean']:>13.3f} +/-{sd:<5.3f}"
            else:
                line += f"{'--':>20}"
        print(line)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
