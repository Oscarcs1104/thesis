"""Property-prediction ablation over PRETRAINED backbones, MoLA fusion, MoleculeNet.

Six configurations x three datasets x N seeds -> one CSV:

    chemberta        ChemBERTa alone            (3 layers, ~3.3M params)
    molformer        MoLFormer-XL alone         (12 layers, hidden 768)
    gin              pretrained GIN alone       the reference the fusion has to beat
    chemberta+gin    both, fused by MoLA cross-layer attention
    molformer+gin    both, fused by MoLA cross-layer attention
    hybrid           the thesis's own HybridMoLA, from scratch

Featurization follows the encoder: the pretrained rows use the Hu et al. 2+2 schema
their GIN checkpoints were trained on, 'hybrid' the OGB 9+3 schema plus character
indices for its SMILES branch. The two are not interchangeable and mixing them raises
nothing.

'hybrid' is also the only row whose checkpoint can go on to initialize the generation
half, since it is the only one sharing that architecture and character vocabulary:

    pretrain_moses.py --arch hybrid  ->  this benchmark (--init-checkpoint)
                                     ->  generation/train_pairs.py (--init-encoder)

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
from typing import Dict, List, Optional

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
from common.wandb_utils import (  # noqa: E402
    add_wandb_args,
    wandb_finish,
    wandb_init,
    wandb_log,
    wandb_summary,
)
from crossmodal_model.model.mola_pretrained import (  # noqa: E402
    CONFIGS,
    build_config,
    precompute_features,
)
from crossmodal_model.train.core import DATASETS  # noqa: E402
from crossmodal_model.generation.pair_data import build_char_vocab  # noqa: E402
from crossmodal_model.model.mola_hybrid import HybridMoLA  # noqa: E402
from data_pipeline.convert_smiles_to_pyg import smiles_to_data  # noqa: E402
from data_pipeline.features_pretrain_gnn import smiles_to_data_pretrain  # noqa: E402

# 'hybrid' is the thesis's own encoder, trained from scratch, and the only row whose
# checkpoint can go on to initialize the generation half: the others use the Hu et al.
# 2+2 featurization and have no character-level SMILES branch.
ALL_CONFIGS = tuple(CONFIGS) + ("hybrid",)

CSV_FIELDS = ["dataset", "config", "seed", "pretrained", "split_protocol", "rmse", "mae", "nrmse", "r2",
              "best_epoch", "n_params", "n_trainable", "elapsed_s"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="*", default=["esol", "freesolv", "lipo"], choices=list(DATASETS))
    p.add_argument("--configs", nargs="*", default=list(ALL_CONFIGS), choices=list(ALL_CONFIGS))
    p.add_argument("--seeds", nargs="*", type=int, default=[2025, 2026, 2027])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--gin-hidden-mult", type=int, default=8,
                   help="GIN MLP inner width as a multiple of hidden_dim. Matches the "
                        "SMILES branch's feedforward block so the modality ablation "
                        "compares modalities rather than capacities. Overridden by "
                        "--init-checkpoint, which fixes the shape")
    p.add_argument("--resplit-per-seed", action="store_true",
                   help="repartition the recombined pool with each run's own seed "
                        "instead of reading the frozen split. The spread across seeds "
                        "then covers the partition as well as the initialisation, which "
                        "is the larger term: repartitioning ESOL moves an unchanged "
                        "model from 0.485 to 0.663 RMSE. Required to compare against "
                        "numbers someone else measured over resampled partitions")
    p.add_argument("--split-strategy", default="deepchem-random",
                   choices=["deepchem-random", "random", "scaffold"],
                   help="which partitioner --resplit-per-seed uses. deepchem-random "
                        "reproduces dc.splits.RandomSplitter exactly, so on the same "
                        "pool with the same seed the partition is identical to one "
                        "drawn with DeepChem -- the comparison becomes molecule-for-"
                        "molecule rather than only in distribution")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--gin-variant", type=str, default="contextpred")
    p.add_argument("--init-checkpoint", type=str, default=None,
                   help="start from a MOSES-pretrained checkpoint "
                        "(crossmodal_model/train/pretrain_moses.py). The head is dropped: it "
                        "predicts 4 standardized RDKit descriptors, not this dataset's target")
    p.add_argument("--freeze-lm", dest="freeze_lm", action="store_true", default=True)
    p.add_argument("--no-freeze-lm", dest="freeze_lm", action="store_false")
    p.add_argument("--freeze-gin", dest="freeze_gin", action="store_true", default=True)
    p.add_argument("--no-freeze-gin", dest="freeze_gin", action="store_false")
    p.add_argument("--no-cache", action="store_true",
                   help="recompute frozen backbones every epoch instead of caching them")
    p.add_argument("--out", type=str, default=str(ROOT / "results" / "pretrained_ablation.csv"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    add_wandb_args(p)
    p.set_defaults(wandb_project="thesis-molnet-finetune")
    return p.parse_args()


def resplit_pool(dataset: str, seed: int, strategy: str = "deepchem-random"):
    """Recombine the three frozen CSVs and repartition them with this run's seed.

    The frozen split answers "is this model better than that model", since every row of
    the table sees identical data. It cannot answer "is this number good", because one
    partition of 113 ESOL test molecules is a single draw: repartitioning the same data
    moves the RMSE of an unchanged model from 0.485 to 0.663. A table built on one
    partition reports a spread that only covers weight initialisation and hides the
    larger term, and its mean cannot be compared against a number someone else measured
    over resampled partitions.

    So both are kept. --resplit-per-seed turns this on; off, the frozen split stands.
    """
    from data_pipeline.splitters import split_dataset

    cfg = DATASETS[dataset]
    csv_dir = ROOT / "data" / "deepchem_molnet" / cfg["dir"] / "csv"
    pool = pd.concat([pd.read_csv(csv_dir / f"{s}.csv") for s in ("train", "valid", "test")],
                     ignore_index=True)
    smiles = pool["smiles"].astype(str).tolist()
    tr, va, te = split_dataset(range(len(pool)), strategy, 0.8, 0.1, 0.1, seed,
                               smiles_list=smiles)
    parts = {"train": list(tr.indices), "valid": list(va.indices), "test": list(te.indices)}
    print(f"    [{dataset} seed {seed}] repartitioned: "
          + " ".join(f"{k}={len(v)}" for k, v in parts.items()))
    return {k: pool.iloc[v].reset_index(drop=True) for k, v in parts.items()}


def load_split(dataset: str, hybrid: bool = False,
               char_vocab: Optional[Dict[str, int]] = None, max_sm_len: int = 100,
               resplit_seed: Optional[int] = None):
    """Featurize a dataset's split, frozen on disk or repartitioned for this seed.

    The schema follows the encoder: 'hybrid' consumes OGB 9+3 plus character indices for
    its SMILES branch, the pretrained-backbone rows the Hu et al. 2+2 schema their GIN
    checkpoints were trained on. Feeding one to the other raises nothing and means
    nothing.
    """
    cfg = DATASETS[dataset]
    csv_dir = ROOT / "data" / "deepchem_molnet" / cfg["dir"] / "csv"
    if resplit_seed is not None:
        raw = resplit_pool(dataset, resplit_seed, strategy=resplit_strategy)
    else:
        raw = {s: pd.read_csv(csv_dir / f"{s}.csv") for s in ("train", "valid", "test")}

    if hybrid and char_vocab is None:
        # Only when there is no pretrained checkpoint to inherit from. Built over all
        # three splits so an unseen character in test is not silently mapped to padding.
        char_vocab = build_char_vocab([s for df in raw.values() for s in df["smiles"].astype(str)])

    featurize = smiles_to_data if hybrid else smiles_to_data_pretrain
    out = {}
    for split, df in raw.items():
        items = []
        for smi, y in zip(df["smiles"].astype(str), df[cfg["target_col"]].astype(float)):
            d = featurize(smi, target=y)
            if d is None:
                continue
            if hybrid:
                ids = [char_vocab.get(c, 0) for c in smi[:max_sm_len]]
                ids.extend([0] * (max_sm_len - len(ids)))
                d.sm = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
            items.append(d)
        out[split] = items
        dropped = len(df) - len(items)
        if dropped:
            print(f"    [{dataset}/{split}] dropped {dropped} unparseable molecules")
    return out, char_vocab


def run_one(dataset: str, config: str, seed: int, args, group: str) -> Dict[str, float]:
    seed_everything(seed, deterministic=False)
    device = args.device
    start = time.time()

    is_hybrid = config == "hybrid"
    init_ckpt = None
    char_vocab = None
    if args.init_checkpoint:
        init_ckpt = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        # The pretrained SMILES embedding is indexed by character id, so reusing the
        # checkpoint's vocabulary is not a convenience -- rebuilding it here would make
        # every row of that embedding stand for a different character.
        char_vocab = init_ckpt.get("char_vocab")

    # Same rule as the generator: the encoder's shape is whatever the checkpoint was
    # trained in, never this script's default.
    hidden_dim, num_layers = args.hidden_dim, args.num_layers
    gin_mult = args.gin_hidden_mult
    if init_ckpt is not None:
        ck_args = init_ckpt.get("args", {})
        hidden_dim = int(ck_args.get("hidden_dim", hidden_dim))
        num_layers = int(ck_args.get("num_layers", num_layers))
        # Same reason as hidden/layers: the pretrained weights only fit the shape they
        # were trained in, and the GIN MLP width is part of that shape.
        gin_mult = int(init_ckpt.get("gin_hidden_mult", ck_args.get("gin_hidden_mult", gin_mult)))
        if (hidden_dim, num_layers) != (args.hidden_dim, args.num_layers):
            print(f"  encoder dims taken from the checkpoint: hidden {hidden_dim}, layers {num_layers}")

    splits, char_vocab = load_split(dataset, hybrid=is_hybrid, char_vocab=char_vocab,
                                    resplit_seed=seed if args.resplit_per_seed else None,
                                    resplit_strategy=args.split_strategy)
    train_y = torch.tensor([float(d.y) for d in splits["train"]])
    standardizer = TargetStandardizer(enabled=True).fit(train_y)
    target_std = float(train_y.std())

    if is_hybrid:
        model = HybridMoLA(
            sm_vocab_size=len(char_vocab), hidden_dim=hidden_dim, output_dim=1,
            num_layers=num_layers, positional_smiles=True, max_sm_len=100,
            gin_hidden_mult=gin_mult,
        ).to(device)
    else:
        model = build_config(
            config, hidden_dim=hidden_dim, output_dim=1, num_layers=num_layers,
            lm_freeze=args.freeze_lm, gin_variant=args.gin_variant, gin_freeze=args.freeze_gin,
        ).to(device)
    if init_ckpt is not None:
        ckpt = init_ckpt
        expected = "hybrid" if is_hybrid else config
        got = ckpt.get("arch") if ckpt.get("arch") == "hybrid" else ckpt.get("config")
        if got != expected:
            raise SystemExit(
                f"--init-checkpoint was pretrained as {got!r} but this row is {expected!r}. "
                f"Loading across configurations would mix backbones silently."
            )
        # HybridMoLA's regression head is out_layer_final; PretrainedMoLA's is head.
        head_prefix = "out_layer_final." if is_hybrid else "head."
        state = {k: v for k, v in ckpt["model_state_dict"].items() if not k.startswith(head_prefix)}
        missing, unexpected = model.load_state_dict(state, strict=False)
        real_missing = [k for k in missing if not k.startswith(head_prefix)]
        if real_missing or unexpected:
            raise SystemExit("pretrained init did not load cleanly\n"
                             f"  missing:    {real_missing}\n"
                             f"  unexpected: {list(unexpected)}")
        print(f"  initialized from {args.init_checkpoint} (step {ckpt.get('step')}), head reset")

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Deterministic loaders: these feed the feature cache, where a reordered or
    # truncated split would silently misalign features and targets.
    ordered = {
        k: GeomDataLoader(v, batch_size=args.batch_size, shuffle=False, drop_last=False)
        for k, v in splits.items()
    }
    # Separate shuffled loader for the uncached path. drop_last guards BatchNorm1d, which
    # cannot compute statistics from a single row, and a trailing batch of one is common
    # on FreeSolv-sized splits.
    loaders = dict(ordered)
    loaders["train"] = GeomDataLoader(
        splits["train"], batch_size=args.batch_size, shuffle=True,
        drop_last=len(splits["train"]) > args.batch_size,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = build_scheduler("plateau", optimizer, args.warmup_epochs, args.epochs, 0.01, 0.5, 5)

    # With every backbone frozen their per-layer states are a deterministic function of
    # the molecule, so they run once here instead of once per epoch. On a 100-epoch run
    # that is the whole cost of the job.
    # Caching only applies to frozen pretrained backbones. HybridMoLA has none -- it is
    # trained end to end -- and has no encode()/fuse() split to replay, so it always takes
    # the direct path.
    cached = None
    if getattr(model, "backbones_frozen", False) and not args.no_cache:
        t0 = time.time()
        cached = {k: precompute_features(model, v, device) for k, v in ordered.items()}
        print(f"  precomputed frozen backbone features in {time.time() - t0:.0f}s", flush=True)

    def _cached_batches(feats, shuffle: bool):
        n = feats["y"].size(0)
        order = torch.randperm(n) if shuffle else torch.arange(n)
        for i in range(0, n, args.batch_size):
            sel = order[i: i + args.batch_size]
            yield (
                feats["graph"][sel].to(device) if "graph" in feats else None,
                feats["lm"][sel].to(device) if "lm" in feats else None,
                feats["y"][sel].to(device).view(-1, 1),
            )

    def epoch_cached(feats, train: bool) -> Dict[str, float]:
        model.train(train)
        preds_all, targets_all, total, n = [], [], 0.0, 0
        for graph, lm, targets in _cached_batches(feats, shuffle=train):
            with torch.set_grad_enabled(train):
                preds = model.fuse(graph, lm)[-1]
                loss = criterion(preds, standardizer.transform(targets))
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            preds_all.append(standardizer.inverse_transform(preds).detach().cpu())
            targets_all.append(targets.detach().cpu())
            total += loss.item() * targets.size(0)
            n += targets.size(0)
        m = regression_metrics(torch.cat(preds_all), torch.cat(targets_all), target_std)
        m["loss"] = total / max(n, 1)
        return m

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

    run = (lambda split, train: epoch_cached(cached[split], train)) if cached is not None         else (lambda split, train: epoch(loaders[split], train))

    # One wandb run per grid cell, all sharing a group: nine scattered runs would be
    # unreadable, and a single run cannot hold nine separate loss curves.
    tag = "pretrained" if args.init_checkpoint else "scratch"
    wb = wandb_init(args, config={**vars(args), "dataset": dataset, "config": config,
                                  "seed": seed, "pretrained": bool(args.init_checkpoint)},
                    name=f"{dataset}_{config}_{tag}_s{seed}", group=group,
                    tags=[dataset, config, tag])

    best_rmse, best_epoch, best_state, stale = float("inf"), 0, None, 0
    for ep in range(1, args.epochs + 1):
        tr = run("train", True)
        val = run("valid", False)
        step_scheduler(scheduler, val["rmse"])
        wandb_log(wb, {"train/rmse": tr["rmse"], "train/loss": tr["loss"],
                       "val/rmse": val["rmse"], "val/mae": val["mae"], "val/r2": val["r2"],
                       "lr": optimizer.param_groups[0]["lr"]}, step=ep)
        if val["rmse"] < best_rmse:
            best_rmse, best_epoch, stale = val["rmse"], ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test = run("test", False)
    wandb_summary(wb, {"test_rmse": test["rmse"], "test_mae": test["mae"],
                       "test_r2": test["r2"], "best_epoch": best_epoch,
                       "n_trainable": n_trainable, "pretrained": bool(args.init_checkpoint)})
    wandb_finish(wb)
    return {
        "dataset": dataset, "config": config, "seed": seed,
        "pretrained": bool(args.init_checkpoint),
        # Which partition protocol produced this row. Two rows measured under different
        # protocols are not comparable, and without this the CSV cannot say which is which.
        "split_protocol": (f"resplit:{args.split_strategy}" if args.resplit_per_seed
                           else "frozen"),
        "rmse": test["rmse"], "mae": test["mae"], "nrmse": test["nrmse"], "r2": test["r2"],
        "best_epoch": best_epoch, "n_params": n_params, "n_trainable": n_trainable,
        "elapsed_s": time.time() - start,
    }


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()

    group = (args.wandb_group
             or f"molnet-{'pretrained' if args.init_checkpoint else 'scratch'}")
    rows: List[Dict[str, float]] = []
    for dataset in args.datasets:
        for config in args.configs:
            for seed in args.seeds:
                print(f"\n=== {dataset} | {config} | seed {seed} ===", flush=True)
                try:
                    row = run_one(dataset, config, seed, args, group)
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
