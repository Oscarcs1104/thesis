"""Supervised pretraining on MOSES: multi-task regression over exact RDKit descriptors.

Everything is UNFROZEN here, on purpose. MoLA's premise is not merely that different
layers carry different information -- it is that the fusion objective *shapes* them that
way. With frozen backbones the cross-layer attention can only select and weight whatever
ChemBERTa happens to emit at each depth; it cannot shape anything, which makes it
attention pooling over frozen features rather than MoLA. Unfreezing is only affordable
because there is enough data here: 1.94M molecules against ~3.3M parameters for
ChemBERTa + 1.9M for the GIN. On 642 FreeSolv molecules it would not be.

The four targets -- logP (Crippen), TPSA, QED, MW -- are an exact oracle, not a model, so
there is no label noise and no ceiling from a teacher. They are also well aligned with
the downstream tasks rather than arbitrary auxiliaries: Crippen logP is the dominant term
in Delaney's ESOL equation, and Lipophilicity's logD 7.4 is logP corrected for ionization.

Known limitation, worth stating in the write-up: MOSES is filtered to drug-like chemical
space, while FreeSolv is mostly small solvents. The pretraining distribution covers that
downstream set poorly, and it is where the transfer should be expected to help least.

    python crossmodal_model/train/pretrain_moses.py --config chemberta+gin
    python crossmodal_model/train/pretrain_moses.py --config molformer+gin --lm-top-n 4
"""
from __future__ import annotations

import argparse
import json
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

from torch_geometric.data import Data  # noqa: E402
from torch_geometric.data import Dataset as GeomDataset  # noqa: E402
from torch_geometric.loader import DataLoader as GeomDataLoader  # noqa: E402

from common.repro import seed_everything  # noqa: E402
from crossmodal_model.generation.pair_data import MoleculeGraphCache, build_char_vocab, cache_name  # noqa: E402
from crossmodal_model.model.mola_hybrid import HybridMoLA  # noqa: E402
from crossmodal_model.model.mola_pretrained import CONFIGS, build_config  # noqa: E402
from data_pipeline.rdkit_labels import PROPERTIES  # noqa: E402


class MosesRegressionDataset(GeomDataset):
    """Graph (Hu et al. schema) + raw SMILES for the LM + the four standardized targets."""

    def __init__(self, cache: MoleculeGraphCache, smiles: List[str],
                 indices: np.ndarray, targets: np.ndarray) -> None:
        super().__init__()
        self.cache = cache
        self.smiles = smiles
        self.indices = indices
        self.targets = torch.from_numpy(targets.astype(np.float32))

    def len(self) -> int:
        return len(self.indices)

    def get(self, idx: int) -> Data:
        i = int(self.indices[idx])
        d = self.cache.get(i)
        d.smiles = self.smiles[i]
        d.y = self.targets[idx].unsqueeze(0)
        return d


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", type=str, default="pretrained", choices=["pretrained", "hybrid"],
                   help="'pretrained' fuses published backbones (Hu GIN + ChemBERTa/MoLFormer); "
                        "'hybrid' is the thesis's own HybridMoLA, trained from scratch. Only the "
                        "hybrid checkpoint can initialize the generation half -- it is the only "
                        "one sharing that architecture, featurization and character vocabulary")
    p.add_argument("--config", type=str, default="chemberta+gin", choices=list(CONFIGS),
                   help="which backbones to fuse; ignored when --arch hybrid")
    p.add_argument("--use-graph", dest="use_graph", action="store_true", default=True)
    p.add_argument("--no-use-graph", dest="use_graph", action="store_false")
    p.add_argument("--use-smiles", dest="use_smiles", action="store_true", default=True)
    p.add_argument("--no-use-smiles", dest="use_smiles", action="store_false")
    p.add_argument("--corpus-dir", type=str, default="data/moses")
    p.add_argument("--max-steps", type=int, default=40000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--eval-batches", type=int, default=50)
    p.add_argument("--ckpt-every-min", type=float, default=30.0)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="low by default: this fine-tunes pretrained backbones, and the "
                        "usual 1e-3 would wash out what they already know in the warmup")
    p.add_argument("--head-lr-mult", type=float, default=10.0,
                   help="the fusion and head start from scratch and need a faster rate "
                        "than the backbones they sit on")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--lm-top-n", type=int, default=0,
                   help="unfreeze only the top N transformer layers of the LM (0 = all). "
                        "MoLFormer is 12 layers at hidden 768; ChemBERTa only 3, so this "
                        "is mostly a MoLFormer knob")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--val-frac", type=float, default=0.01)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--out-dir", type=str, default="checkpoints/pretrain_moses")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def restrict_lm_to_top_n(model, top_n: int) -> None:
    """Freeze all but the top N transformer layers of the language backbone.

    Layer modules are found by name rather than by a fixed attribute path, because
    ChemBERTa is a RoBERTa and MoLFormer ships its own custom architecture.
    """
    if top_n <= 0 or not getattr(model, "use_lm", False):
        return
    backbone = model.lm.backbone
    layer_lists = [m for name, m in backbone.named_modules()
                   if isinstance(m, nn.ModuleList) and len(m) >= top_n]
    if not layer_lists:
        print(f"  [warn] could not locate the layer stack in {model.lm.model_name}; "
              f"leaving the whole backbone trainable")
        return
    stack = max(layer_lists, key=len)
    for p in backbone.parameters():
        p.requires_grad = False
    for layer in list(stack)[-top_n:]:
        for p in layer.parameters():
            p.requires_grad = True
    print(f"  LM: unfroze the top {top_n} of {len(stack)} layers")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed, deterministic=False)
    device = args.device
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.arch == "hybrid":
        arm = ("graph+smiles" if args.use_graph and args.use_smiles
               else "graph-only" if args.use_graph else "smiles-only")
        tag = f"hybrid_{arm}"
    else:
        tag = args.config.replace("+", "_")
    run_name = f"{tag}_s{args.seed}"
    ckpt_path = out_dir / f"{run_name}.pt"

    if device.startswith("cuda"):
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} | VRAM {props.total_memory / 1e9:.0f} GB")

    corpus_dir = ROOT / args.corpus_dir
    corpus = pd.read_csv(corpus_dir / "corpus.csv")
    labels = np.load(corpus_dir / "labels.npy")
    if args.limit:
        corpus, labels = corpus.iloc[:args.limit], labels[:args.limit]
    smiles = corpus["smiles"].astype(str).tolist()

    # The Hu et al. schema, kept in its own cache file: it is not interchangeable with
    # the OGB one the generation half uses.
    schema = "ogb" if args.arch == "hybrid" else "pretrain-gnn"
    cache_path = corpus_dir / cache_name(schema)
    if cache_path.exists() and not args.rebuild_cache:
        cache = MoleculeGraphCache.load(cache_path)
        print(f"Loaded graph cache from {cache_path} ({len(cache):,} molecules)")
    else:
        cache = MoleculeGraphCache.build(smiles, build_char_vocab(smiles),
                                         workers=args.num_workers * 2, schema=schema)
        cache.save(cache_path)
        print(f"Saved graph cache to {cache_path}")
    if cache.schema != schema:
        raise SystemExit(f"cache at {cache_path} uses the {cache.schema!r} schema but "
                         f"--arch {args.arch} needs {schema!r}. Pass --rebuild-cache.")

    usable = np.flatnonzero(np.isfinite(labels).all(axis=1) & cache.valid[:len(labels)])
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(usable)
    n_val = int(len(perm) * args.val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    # Standardized per property: MW has a std near 28 and QED near 0.1, so an unweighted
    # MSE over raw targets would be a molecular-weight regressor with three decorations.
    mean = labels[train_idx].mean(axis=0)
    std = labels[train_idx].std(axis=0)
    std[std == 0] = 1.0
    z = (labels - mean) / std
    print(f"Train {len(train_idx):,} | val {len(val_idx):,}")
    for p, name in enumerate(PROPERTIES):
        print(f"  {name:<6} mean {mean[p]:>9.3f}  std {std[p]:>8.3f}")

    train_ds = MosesRegressionDataset(cache, smiles, train_idx, z[train_idx])
    val_ds = MosesRegressionDataset(cache, smiles, val_idx, z[val_idx])
    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                         pin_memory=device.startswith("cuda"),
                         persistent_workers=args.num_workers > 0)
    train_loader = GeomDataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = GeomDataLoader(val_ds, shuffle=False, **loader_kwargs)

    if args.arch == "hybrid":
        model = HybridMoLA(
            sm_vocab_size=len(cache.char_vocab), hidden_dim=args.hidden_dim,
            output_dim=len(PROPERTIES), num_layers=args.num_layers,
            positional_smiles=True, max_sm_len=cache.sm.shape[1],
            use_graph=args.use_graph, use_smiles=args.use_smiles,
        ).to(device)
        print(f"  arch=hybrid ({tag}), everything from scratch")
    else:
        model = build_config(
            args.config, hidden_dim=args.hidden_dim, output_dim=len(PROPERTIES),
            num_layers=args.num_layers, lm_freeze=False, gin_freeze=False,
        ).to(device)
        restrict_lm_to_top_n(model, args.lm_top_n)

    # Two rates only when there is a pretrained backbone to protect. With everything
    # from scratch a split would just slow half the model down for no reason.
    backbone_names = ("gin.", "lm.backbone.")
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_backbone = args.arch != "hybrid" and name.startswith(backbone_names)
        (backbone_params if is_backbone else head_params).append(param)
    print(f"  trainable: {sum(p.numel() for p in backbone_params):,} backbone + "
          f"{sum(p.numel() for p in head_params):,} fusion/head")

    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": args.lr},
         {"params": head_params, "lr": args.lr * args.head_lr_mult}],
        weight_decay=args.weight_decay, fused=device.startswith("cuda"),
    )
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    criterion = nn.MSELoss()
    use_amp = device.startswith("cuda") and torch.cuda.is_bf16_supported()
    if use_amp:
        print("  bf16 autocast enabled")

    def set_lr(step: int) -> None:
        if step < args.warmup_steps:
            scale = step / max(args.warmup_steps, 1)
        else:
            progress = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
            scale = max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
        for g, base in zip(optimizer.param_groups, base_lrs):
            g["lr"] = base * scale

    @torch.no_grad()
    def evaluate() -> Dict[str, float]:
        model.eval()
        preds, trues = [], []
        for i, batch in enumerate(val_loader):
            if i >= args.eval_batches:
                break
            batch = batch.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = model(batch)[-1]
            preds.append(out.float().cpu())
            trues.append(batch.y.view_as(out).float().cpu())
        model.train()
        p, t = torch.cat(preds), torch.cat(trues)
        # Reported in each property's own units, not in standardized space, so the
        # numbers mean something chemically.
        rmse = {name: float(((p[:, i] - t[:, i]) ** 2).mean().sqrt() * std[i])
                for i, name in enumerate(PROPERTIES)}
        return {"loss": float(((p - t) ** 2).mean()), **rmse}

    step, history, best = 0, [], float("inf")
    start = last_ckpt = time.time()
    model.train()
    print()
    while step < args.max_steps:
        for batch in train_loader:
            if step >= args.max_steps:
                break
            batch = batch.to(device, non_blocking=True)
            set_lr(step)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = model(batch)[-1]
                loss = criterion(out.float(), batch.y.view_as(out).float())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            step += 1

            if step % args.eval_every == 0 or step == args.max_steps:
                val = evaluate()
                rate = step / max(time.time() - start, 1e-9)
                units = "  ".join(f"{k} {val[k]:.3f}" for k in PROPERTIES)
                print(f"step {step:>7,}/{args.max_steps:,} | train {loss.item():.4f} | "
                      f"val {val['loss']:.4f} | RMSE {units} | "
                      f"{rate:.1f} it/s | eta {(args.max_steps - step) / max(rate, 1e-9) / 60:.0f} min",
                      flush=True)
                history.append({"step": step, "train_loss": loss.item(), **val})
                best = min(best, val["loss"])

            if time.time() - last_ckpt > args.ckpt_every_min * 60 or step == args.max_steps:
                torch.save({
                    "model_state_dict": model.state_dict(), "config": args.config,
                    "arch": args.arch, "char_vocab": cache.char_vocab, "schema": schema,
                    "use_graph": args.use_graph, "use_smiles": args.use_smiles,
                    "step": step, "args": vars(args), "history": history,
                    # Needed by the fine-tune: without them the pretrained head predicts
                    # in standardized space and its outputs are meaningless.
                    "target_mean": mean.tolist(), "target_std": std.tolist(),
                    "properties": PROPERTIES,
                }, ckpt_path)
                last_ckpt = time.time()

    print(f"\nDone: {step:,} steps in {(time.time() - start) / 60:.0f} min | best val {best:.4f}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"\nFine-tune with:\n  python crossmodal_model/benchmark/pretrained_ablation.py "
          f"--configs {args.config} --init-checkpoint {ckpt_path}")
    (out_dir / f"{run_name}_history.json").write_text(
        json.dumps({"config": args.config, "history": history}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
