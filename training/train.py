"""Training entrypoint for the multimodal **property predictor**. Regression only.

Generation lives in `training/train_generator.py` (a standalone conditional
SELFIES model), not here -- see docs/evaluation.md for why they are decoupled.

Design points carried over from the audit refactor:
  B1  frozen LM stays in eval() (handled in the model's train() override)
  B2  the NRMSE denominator (train target std) is computed on the TRAIN split only
  B7  full seeding (torch / cuda / numpy / random) via seed_everything
  D1  split strategy is explicit: use a dataset's OFFICIAL predefined
      train/val/test split when one exists (--dataset-dir / --train-path...),
      otherwise fall back to an internal --split {scaffold,random} on --data-path.
      The thesis uses the DeepChem *scaffold* split for all three datasets.
  D3  target standardization fit on TRAIN targets only, inverted for metrics
  D4  warmup -> plateau/cosine LR schedule, grad-norm clipping, multi-seed --seeds
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader as GeomDataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data_pipeline.convert_smiles_to_pyg import randomize_smiles
from data_pipeline.data import HybridGraphLangDataset, load_graph_dataset
from data_pipeline.splitters import split_dataset as split_dataset_by_strategy
from model.model import build_model_from_args
from training.repro import (
    TargetStandardizer,
    aggregate_seed_metrics,
    build_scheduler,
    format_seed_table,
    regression_metrics,
    seed_everything,
    step_scheduler,
)
from training.wandb_utils import add_wandb_args, wandb_finish, wandb_init, wandb_log


def _augment_smiles_batch(smiles_batch, augment_prob: float):
    if augment_prob <= 0:
        return smiles_batch
    return [randomize_smiles(smi) if random.random() < augment_prob else smi for smi in smiles_batch]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the multimodal property predictor (regression)")
    parser.add_argument("--data-path", type=str, default=None, help="Single CSV/graph dataset, split internally via --split. Ignored if a predefined split is given.")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Folder with csv/{train,valid,test}.csv (from data_pipeline/prepare_all.py) -- uses that official split as-is (D1).")
    parser.add_argument("--train-path", type=str, default=None)
    parser.add_argument("--val-path", type=str, default=None)
    parser.add_argument("--test-path", type=str, default=None)
    parser.add_argument("--output-dim", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--graph-backbone", type=str, default="gin", choices=["gcn", "gat", "gatv2", "gin"])
    parser.add_argument("--use-graph", action=argparse.BooleanOptionalAction, default=True, help="--no-use-graph for a language-only ablation")
    parser.add_argument("--freeze-graph-encoder", action=argparse.BooleanOptionalAction, default=False, help="Freeze the graph encoder (e.g. as a fixed feature extractor with --graph-pretrained-checkpoint)")
    parser.add_argument("--language-backbone", type=str, default="huggingface", choices=["huggingface", "none"])
    parser.add_argument("--language-model-name", type=str, default="DeepChem/ChemBERTa-77M-MLM", help="Any HuggingFace text-encoder repo id")
    parser.add_argument("--freeze-language-backbone", action=argparse.BooleanOptionalAction, default=True, help="--no-freeze-language-backbone fine-tunes the whole text backbone (D2)")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--use-language", action=argparse.BooleanOptionalAction, default=True, help="--no-use-language for a graph-only ablation")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=0, help="If >0, torch.set_num_threads(N).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--lr-schedule", type=str, default="plateau", choices=["plateau", "cosine"])
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-patience", type=int, default=5)
    parser.add_argument("--min-lr-ratio", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--split", type=str, default="scaffold", choices=["scaffold", "random"], help="Split strategy for --data-path ONLY -- ignored when a predefined split is given (D1)")
    parser.add_argument("--standardize-target", action=argparse.BooleanOptionalAction, default=True, help="Standardize the target on the TRAIN split only (D3)")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="Run once per seed, report test mean +/- std (D4). Overrides --seed.")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--load-checkpoint", type=str, default=None)
    parser.add_argument("--graph-pretrained-checkpoint", type=str, default=None)
    parser.add_argument("--smiles-augment-prob", type=float, default=0.0, help="Prob. of feeding a randomized (non-canonical) SMILES to the language branch (train only).")
    parser.add_argument("--linear-probe-epochs", type=int, default=0, help="With --graph-pretrained-checkpoint: freeze the graph encoder for N epochs (train only the head), then unfreeze at --unfreeze-lr-mult * --lr.")
    parser.add_argument("--unfreeze-lr-mult", type=float, default=0.1)
    add_wandb_args(parser)
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int = 0) -> GeomDataLoader:
    # drop_last only on the shuffled (training) loader, and only if it wouldn't drop
    # everything: a size-1 trailing batch of a lone single-atom molecule (FreeSolv has
    # "C"/"N"/"S") makes BatchNorm1d in the GIN-E MLP raise. Re-shuffled each epoch, so
    # nothing is permanently excluded.
    drop_last = shuffle and len(dataset) > batch_size
    return GeomDataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
        num_workers=num_workers, persistent_workers=num_workers > 0,
    )


def resolve_predefined_split(args: argparse.Namespace) -> Optional[Tuple[str, str, str]]:
    """D1: prefer a dataset's own official split over re-splitting it ourselves."""
    explicit = (args.train_path, args.val_path, args.test_path)
    if any(explicit):
        if not all(explicit):
            raise ValueError("--train-path, --val-path and --test-path must all be given together")
        return explicit  # type: ignore[return-value]
    if args.dataset_dir:
        base = Path(args.dataset_dir) / "csv"
        train_p, val_p, test_p = base / "train.csv", base / "valid.csv", base / "test.csv"
        missing = [str(p) for p in (train_p, val_p, test_p) if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing predefined split file(s): {missing}. Run data_pipeline/prepare_all.py first.")
        return str(train_p), str(val_p), str(test_p)
    return None


def load_predefined_datasets(train_path: str, val_path: str, test_path: str):
    return (
        HybridGraphLangDataset(load_graph_dataset(train_path)),
        HybridGraphLangDataset(load_graph_dataset(val_path)),
        HybridGraphLangDataset(load_graph_dataset(test_path)),
    )


def _subset_targets(subset) -> torch.Tensor:
    rows: List[torch.Tensor] = []
    for i in range(len(subset)):
        y = getattr(subset[i], "y", None)
        if y is None:
            continue
        rows.append(torch.as_tensor(y).float().view(-1))
    if not rows:
        return torch.zeros(0, 1)
    width = max(r.numel() for r in rows)
    return torch.stack([r if r.numel() == width else r.view(-1)[:width] for r in rows])


def _target_std(train_subset) -> float:
    """B2: NRMSE denominator = std of the TRAIN targets only (0.0 -> NRMSE is nan)."""
    y = _subset_targets(train_subset).view(-1)
    if y.numel() < 2:
        return 0.0
    return float(y.std(unbiased=True))


def _print_target_stats(label: str, train_subset) -> None:
    y = _subset_targets(train_subset).view(-1)
    if y.numel() == 0:
        return
    print(f"[{label}] train target stats: n={y.numel()} mean={y.mean():.3f} std={y.std():.3f} range=[{y.min():.3f}, {y.max():.3f}]")


# --------------------------------------------------------------------------- #
# Eval / train loops (predictor only)
# --------------------------------------------------------------------------- #
def evaluate(model, loader, criterion, device, target_std=None, standardizer: Optional[TargetStandardizer] = None,
             return_arrays: bool = False) -> Dict[str, float]:
    model.eval()
    total_loss, total_items = 0.0, 0
    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    all_smiles: List[str] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            targets_orig = batch.y.float().view_as(logits)
            targets_std = standardizer.transform(targets_orig) if standardizer is not None else targets_orig
            loss = criterion(logits, targets_std)
            preds_orig = standardizer.inverse_transform(logits) if standardizer is not None else logits
            all_preds.append(preds_orig.detach().cpu().view(-1))
            all_targets.append(targets_orig.detach().cpu().view(-1))
            batch_smiles = getattr(batch, "smiles", None)
            if batch_smiles is not None:
                all_smiles.extend(str(s) for s in batch_smiles)
            total_loss += loss.item() * batch.num_graphs
            total_items += batch.num_graphs
    metrics = {"loss": total_loss / max(total_items, 1)}
    preds = torch.cat(all_preds) if all_preds else torch.zeros(0)
    targets = torch.cat(all_targets) if all_targets else torch.zeros(0)
    metrics.update(regression_metrics(preds, targets, target_std))
    if return_arrays:
        metrics["y_true"] = targets
        metrics["y_pred"] = preds
        metrics["smiles"] = all_smiles or None
    return metrics


def train_one_epoch(model, loader, optimizer, criterion, device, target_std, smiles_augment_prob: float = 0.0,
                    standardizer: Optional[TargetStandardizer] = None, grad_clip: float = 0.0) -> Dict[str, float]:
    model.train()
    total_loss, total_items = 0.0, 0
    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        if smiles_augment_prob > 0 and getattr(batch, "smiles", None) is not None:
            batch.smiles = _augment_smiles_batch(list(batch.smiles), smiles_augment_prob)
        logits = model(batch)
        targets_orig = batch.y.float().view_as(logits)
        targets_std = standardizer.transform(targets_orig) if standardizer is not None else targets_orig
        loss = criterion(logits, targets_std)
        preds_orig = standardizer.inverse_transform(logits) if standardizer is not None else logits
        all_preds.append(preds_orig.detach().cpu())
        all_targets.append(targets_orig.detach().cpu())
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        total_items += batch.num_graphs
    metrics = {"loss": total_loss / max(total_items, 1)}
    if all_preds:
        metrics.update(regression_metrics(torch.cat(all_preds), torch.cat(all_targets), target_std))
    return metrics


def load_graph_pretrained_checkpoint(model, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("graph_encoder", checkpoint.get("encoder_state_dict", checkpoint.get("model_state_dict", {})))
    if not isinstance(state_dict, dict):
        return
    model_state = model.graph_encoder.state_dict()
    mismatched = [k for k in state_dict if k in model_state and tuple(state_dict[k].shape) != tuple(model_state[k].shape)]
    if mismatched:
        ex = mismatched[0]
        raise ValueError(
            f"--graph-pretrained-checkpoint {checkpoint_path!r} has a different graph_encoder shape than this run "
            f"(--hidden-dim / --num-layers / --graph-backbone must match the pretraining run). "
            f"E.g. {ex}: checkpoint {tuple(state_dict[ex].shape)} vs model {tuple(model_state[ex].shape)}. "
            f"({len(mismatched)} tensor(s) mismatched.)"
        )
    model.graph_encoder.load_state_dict(state_dict, strict=False)


# --------------------------------------------------------------------------- #
# Shared predictor loop used by the fusion-ablation trainers (CA / MoE)
# --------------------------------------------------------------------------- #
def run_predictor_ablation_training(model, args, train_set, val_set, test_set, target_std, criterion,
                                    predictions_out: Optional[str] = None) -> dict:
    device = args.device
    standardize = getattr(args, "standardize_target", True)
    standardizer = TargetStandardizer(enabled=standardize).fit(_subset_targets(train_set)) if standardize else None
    if standardizer is not None:
        _print_target_stats("ablation", train_set)
    grad_clip = getattr(args, "grad_clip", 1.0)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(
        getattr(args, "lr_schedule", "plateau"), optimizer, getattr(args, "warmup_epochs", 5), args.epochs,
        getattr(args, "min_lr_ratio", 0.01), getattr(args, "plateau_factor", 0.5), getattr(args, "plateau_patience", 5),
    )
    wandb_run = wandb_init(args, config=vars(args))
    train_loader = make_loader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(val_set, args.batch_size, shuffle=False, num_workers=args.num_workers)

    best_val_loss, best_state, no_improve = float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, target_std, standardizer=standardizer, grad_clip=grad_clip)
        val_metrics = evaluate(model, val_loader, criterion, device, target_std, standardizer=standardizer)
        step_scheduler(scheduler, val_metrics["loss"])
        print(f"Epoch {epoch:03d} | lr={optimizer.param_groups[0]['lr']:.2e} | train_loss={train_metrics['loss']:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} | train_rmse={train_metrics.get('rmse', float('nan')):.4f} | val_rmse={val_metrics.get('rmse', float('nan')):.4f}")
        wandb_log(wandb_run, {**{f"train/{k}": v for k, v in train_metrics.items()}, **{f"val/{k}": v for k, v in val_metrics.items()}, "lr": optimizer.param_groups[0]["lr"]}, step=epoch)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss, no_improve = val_metrics["loss"], 0
            best_state = {
                "model_state_dict": model.state_dict(), "epoch": epoch, "args": vars(args),
                "target_standardizer": standardizer.state_dict() if standardizer is not None else None,
            }
            if args.checkpoint_path:
                cp = Path(args.checkpoint_path)
                cp.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, cp)
        else:
            no_improve += 1
        if no_improve >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])
    test_loader = make_loader(test_set, args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_metrics = evaluate(model, test_loader, criterion, device, target_std, standardizer=standardizer,
                            return_arrays=bool(predictions_out))
    if predictions_out:
        # dump BOTH val and test predictions so a stacked ensemble can train its
        # meta-learner on val (see baselines/stacking.py)
        val_arrays = evaluate(model, val_loader, criterion, device, target_std, standardizer=standardizer, return_arrays=True)
        Path(predictions_out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "test": {"y_true": test_metrics.pop("y_true"), "y_pred": test_metrics.pop("y_pred"), "smiles": test_metrics.pop("smiles")},
                "val": {"y_true": val_arrays["y_true"], "y_pred": val_arrays["y_pred"], "smiles": val_arrays["smiles"]},
            },
            predictions_out,
        )
    print(f"Test loss={test_metrics['loss']:.4f} | Test RMSE={test_metrics.get('rmse', float('nan')):.4f} | Test NRMSE={test_metrics.get('nrmse', float('nan')):.4f}")
    wandb_log(wandb_run, {f"test/{k}": v for k, v in test_metrics.items() if isinstance(v, (int, float))})
    wandb_finish(wandb_run)
    return test_metrics


# --------------------------------------------------------------------------- #
# One full train/val/test run for a single seed
# --------------------------------------------------------------------------- #
def run_single(args: argparse.Namespace, seed: int, dataset=None, train_set=None, val_set=None, test_set=None) -> Dict[str, float]:
    seed_everything(seed, deterministic=args.deterministic)

    if train_set is None:
        if dataset is None:
            raise ValueError("run_single needs dataset= (internal split) or train_set/val_set/test_set= (predefined split)")
        all_smiles = [str(getattr(dataset[i], "smiles", "")) for i in range(len(dataset))]
        train_set, val_set, test_set = split_dataset_by_strategy(
            dataset, args.split, args.train_ratio, args.val_ratio, args.test_ratio, seed, smiles_list=all_smiles
        )
        print(f"[seed {seed}] internal split={args.split} -> train={len(train_set)} val={len(val_set)} test={len(test_set)}")
    else:
        print(f"[seed {seed}] predefined split -> train={len(train_set)} val={len(val_set)} test={len(test_set)}")

    criterion = nn.MSELoss()
    standardizer = TargetStandardizer(enabled=args.standardize_target).fit(_subset_targets(train_set)) if args.standardize_target else None
    target_std = _target_std(train_set)
    _print_target_stats(f"seed {seed}", train_set)

    model = build_model_from_args(args).to(args.device)
    if args.load_checkpoint:
        ckpt = torch.load(args.load_checkpoint, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict"))
        if state_dict is None:
            raise ValueError("Checkpoint has no model_state_dict / state_dict")
        model.load_state_dict(state_dict, strict=False)
    if args.graph_pretrained_checkpoint:
        load_graph_pretrained_checkpoint(model, args.graph_pretrained_checkpoint)
    if args.freeze_graph_encoder:
        for p in model.graph_encoder.parameters():
            p.requires_grad_(False)

    # Linear probing: freeze the pretrained graph encoder for --linear-probe-epochs
    # (head settles first), then unfreeze at a reduced LR. Both param groups exist
    # up front; unfreezing is just a requires_grad flip -- no optimizer/scheduler rebuild.
    linear_probe_epochs = args.linear_probe_epochs if (args.graph_pretrained_checkpoint and not args.freeze_graph_encoder) else 0
    encoder_params: List[torch.nn.Parameter] = []
    if linear_probe_epochs > 0:
        encoder_params = [p for p in model.graph_encoder.parameters() if p.requires_grad]
        encoder_param_ids = {id(p) for p in encoder_params}
        head_params = [p for p in model.parameters() if p.requires_grad and id(p) not in encoder_param_ids]
        for p in encoder_params:
            p.requires_grad_(False)
        optimizer = torch.optim.AdamW(
            [{"params": head_params, "lr": args.lr},
             {"params": encoder_params, "lr": args.lr * args.unfreeze_lr_mult}],
            weight_decay=args.weight_decay,
        )
        print(f"[seed {seed}] linear probe: graph_encoder frozen for {linear_probe_epochs} epoch(s), then lr={args.lr * args.unfreeze_lr_mult:.2e}")
    else:
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(args.lr_schedule, optimizer, args.warmup_epochs, args.epochs, args.min_lr_ratio, args.plateau_factor, args.plateau_patience)

    run_name = f"{args.wandb_run_name}-s{seed}" if args.wandb_run_name else None
    wandb_run = wandb_init(argparse.Namespace(**{**vars(args), "wandb_run_name": run_name}), config={**vars(args), "seed": seed})
    train_loader = make_loader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(val_set, args.batch_size, shuffle=False, num_workers=args.num_workers)

    best_val_loss, best_state, no_improve = float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        if linear_probe_epochs > 0 and epoch == linear_probe_epochs + 1:
            for p in encoder_params:
                p.requires_grad_(True)
            print(f"[seed {seed}] linear probe done -> unfreezing graph_encoder at epoch {epoch}")
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, args.device, target_std,
                                        args.smiles_augment_prob, standardizer=standardizer, grad_clip=args.grad_clip)
        val_metrics = evaluate(model, val_loader, criterion, args.device, target_std, standardizer=standardizer)
        step_scheduler(scheduler, val_metrics["loss"])
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d} | lr={lr_now:.2e} | train_loss={train_metrics['loss']:.4f} | val_loss={val_metrics['loss']:.4f} | "
              f"train_rmse={train_metrics.get('rmse', float('nan')):.4f} | val_rmse={val_metrics.get('rmse', float('nan')):.4f} | val_nrmse={val_metrics.get('nrmse', float('nan')):.4f}")
        wandb_log(wandb_run, {**{f"train/{k}": v for k, v in train_metrics.items()}, **{f"val/{k}": v for k, v in val_metrics.items()}, "lr": lr_now}, step=epoch)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss, no_improve = val_metrics["loss"], 0
            best_state = {
                "model_state_dict": model.state_dict(), "epoch": epoch, "args": vars(args), "seed": seed,
                "target_standardizer": standardizer.state_dict() if standardizer is not None else None,
            }
            cp = Path(args.checkpoint_path) if args.checkpoint_path else Path("checkpoints") / "best.pt"
            cp = cp.with_name(f"{cp.stem}_s{seed}{cp.suffix}") if args.seeds and len(args.seeds) > 1 else cp
            cp.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, cp)
        else:
            no_improve += 1
        if no_improve >= args.patience:
            print(f"[seed {seed}] early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])
    test_loader = make_loader(test_set, args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_metrics = evaluate(model, test_loader, criterion, args.device, target_std, standardizer=standardizer)
    print(f"[seed {seed}] Test loss={test_metrics['loss']:.4f} | RMSE={test_metrics.get('rmse', float('nan')):.4f} | "
          f"NRMSE={test_metrics.get('nrmse', float('nan')):.4f} | MAE={test_metrics.get('mae', float('nan')):.4f}")
    wandb_log(wandb_run, {f"test/{k}": v for k, v in test_metrics.items()})
    wandb_finish(wandb_run)
    return test_metrics


def main() -> None:
    args = parse_args()
    if args.num_threads and args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    seeds = args.seeds if args.seeds else [args.seed]
    per_seed: List[Dict[str, float]] = []

    predefined = resolve_predefined_split(args)
    if predefined is not None:
        train_set, val_set, test_set = load_predefined_datasets(*predefined)
        print(f"Loaded predefined split from {predefined}: train={len(train_set)} val={len(val_set)} test={len(test_set)}")
        for seed in seeds:
            per_seed.append(run_single(args, seed, train_set=train_set, val_set=val_set, test_set=test_set))
    else:
        if not args.data_path:
            raise ValueError("Provide --dataset-dir (predefined split) or --data-path (internal --split).")
        dataset = HybridGraphLangDataset(load_graph_dataset(args.data_path))
        for seed in seeds:
            per_seed.append(run_single(args, seed, dataset=dataset))

    if len(seeds) > 1:
        print(format_seed_table(aggregate_seed_metrics(per_seed)))


if __name__ == "__main__":
    main()
