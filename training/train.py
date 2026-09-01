"""Training entrypoint for the thesis multimodal model. Regression only.

FASE 1 refactor (audit fixes):
  B1  frozen LM stays in eval() -> handled in model.train() overrides
  B2  target range (NRMSE denominator) computed on the TRAIN split only
  B6  torch.set_num_threads is now the opt-in --num-threads flag
  B7  full seeding (torch / cuda / numpy / random) via seed_everything
  D1  split strategy is explicit: reuse a dataset's OFFICIAL train/val/test split
      when one exists (--dataset-dir / --train-path,--val-path,--test-path,
      comparable to literature numbers), otherwise fall back to an internal
      --split {scaffold,random} on a single --data-path
  D3  target standardization fit on TRAIN targets only, inverted for metrics
  D4  warmup + cosine LR schedule, grad-norm clipping, multi-seed --seeds

Classification support was removed on purpose: this project only benchmarks
regression datasets (ESOL, FreeSolv, Lipophilicity). See training/repro.py for
the (regression-only) metrics helpers.
"""

from __future__ import annotations

import argparse
import math
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
from model.smiles_decoder import build_vocab as build_smiles_vocab, encode_batch as encode_smiles_batch
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
    parser = argparse.ArgumentParser(description="Train the thesis multimodal predictor (regression)")
    parser.add_argument("--data-path", type=str, default=None, help="Single CSV/graph dataset, split internally via --split. Ignored if --dataset-dir or --train-path/--val-path/--test-path is given.")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Folder with csv/train.csv, csv/valid.csv, csv/test.csv (as produced by data_pipeline/download_deepchem_datasets.py) -- uses that official split as-is, no re-splitting (D1).")
    parser.add_argument("--train-path", type=str, default=None, help="Explicit predefined train split (use with --val-path/--test-path)")
    parser.add_argument("--val-path", type=str, default=None)
    parser.add_argument("--test-path", type=str, default=None)
    parser.add_argument("--output-dim", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--graph-backbone", type=str, default="gatv2", choices=["gcn", "gat", "gatv2", "gin"])
    parser.add_argument("--use-graph", action=argparse.BooleanOptionalAction, default=True, help="Disable to predict/generate from the language branch alone (e.g. --no-use-graph for a lang-only ablation)")
    parser.add_argument("--freeze-graph-encoder", action=argparse.BooleanOptionalAction, default=False, help="Freeze the graph encoder's weights (e.g. combine with --graph-pretrained-checkpoint to use it as a fixed, untrained-further feature extractor)")
    parser.add_argument("--language-backbone", type=str, default="huggingface", choices=["huggingface", "none"])
    parser.add_argument("--language-model-name", type=str, default="DeepChem/ChemBERTa-77M-MLM", help="Any HuggingFace text-encoder repo id (e.g. a ChemBERTa or MoLFormer checkpoint)")
    parser.add_argument("--freeze-language-backbone", action=argparse.BooleanOptionalAction, default=True, help="--no-freeze-language-backbone fine-tunes the whole text backbone end to end (D2)")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to transformers (needed by some HF repos, e.g. MoLFormer)")
    parser.add_argument("--use-language", action=argparse.BooleanOptionalAction, default=True, help="Disable for a graph-only ablation")
    parser.add_argument("--node-encoding", type=str, default="dense", choices=["categorical", "dense"])
    parser.add_argument("--node-vocab-sizes", type=int, nargs="*", default=[119, 4])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0, help="Parallel CPU processes for batch loading (e.g. 4 if your machine has 4 CPUs)")
    parser.add_argument("--num-threads", type=int, default=0, help="If >0, torch.set_num_threads(N). 0 leaves PyTorch's default (was hard-coded to 1 -- B6).")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Linear LR warmup epochs before the main schedule kicks in (D4)")
    parser.add_argument("--lr-schedule", type=str, default="plateau", choices=["plateau", "cosine"], help="'plateau' (default) halves lr when val loss stalls -- composes correctly with early stopping. 'cosine' decays over a fixed --epochs horizon regardless of when/if early stopping fires.")
    parser.add_argument("--plateau-factor", type=float, default=0.5, help="LR multiplier on a val-loss plateau (--lr-schedule plateau)")
    parser.add_argument("--plateau-patience", type=int, default=5, help="Epochs without val-loss improvement before the LR is cut (--lr-schedule plateau)")
    parser.add_argument("--min-lr-ratio", type=float, default=0.01, help="LR floor as a fraction of --lr (D4)")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Max grad-norm for clip_grad_norm_ (0 disables) (D4)")
    parser.add_argument("--split", type=str, default="scaffold", choices=["scaffold", "random"], help="Split strategy for --data-path ONLY -- ignored when a predefined split is given (D1)")
    parser.add_argument("--standardize-target", action=argparse.BooleanOptionalAction, default=True, help="Standardize the regression target on the TRAIN split only (D3)")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="Run once per seed and report test mean +/- std (D4). Overrides --seed when given. With a predefined split the split itself doesn't change across seeds -- only model init / loader shuffling do.")
    parser.add_argument("--deterministic", action="store_true", help="Also set cudnn.deterministic (slower)")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--load-checkpoint", type=str, default=None)
    parser.add_argument("--graph-pretrained-checkpoint", type=str, default=None)
    parser.add_argument("--use-decoder", action="store_true")
    parser.add_argument("--training-mode", type=str, default="predictor", choices=["predictor", "joint", "decoder"])
    parser.add_argument("--decoder-loss-weight", type=float, default=0.5)
    parser.add_argument("--decoder-max-len", type=int, default=96)
    parser.add_argument(
        "--smiles-augment-prob",
        type=float,
        default=0.0,
        help="Probability of replacing a molecule's SMILES with a randomized (non-canonical) equivalent before "
        "it reaches the language branch. The decoder target (when --use-decoder) stays canonical regardless.",
    )
    parser.add_argument(
        "--property-context-dropout-prob",
        type=float,
        default=0.0,
        help="Probability, per training sample, of zeroing the molecule-derived context fed to the decoder "
        "(keeping property_values and the true decoder target unchanged).",
    )
    parser.add_argument(
        "--linear-probe-epochs",
        type=int,
        default=0,
        help="Only with --graph-pretrained-checkpoint: freeze the graph_encoder for this many epochs "
        "(training only the head) before unfreezing it with --unfreeze-lr-mult * --lr. Guards a pretrained "
        "encoder against being scrambled by large early gradients from a randomly-initialized head.",
    )
    parser.add_argument("--unfreeze-lr-mult", type=float, default=0.1, help="graph_encoder LR = --lr * this, once --linear-probe-epochs unfreezes it")
    add_wandb_args(parser)
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int = 0) -> GeomDataLoader:
    return GeomDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
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
        missing = [p for p in (train_p, val_p, test_p) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing predefined split file(s): {[str(p) for p in missing]}. "
                f"Run data_pipeline/download_deepchem_datasets.py --datasets <name> --output-dir <parent of --dataset-dir> first."
            )
        return str(train_p), str(val_p), str(test_p)
    return None


def load_predefined_datasets(train_path: str, val_path: str, test_path: str) -> Tuple[HybridGraphLangDataset, HybridGraphLangDataset, HybridGraphLangDataset]:
    return (
        HybridGraphLangDataset(load_graph_dataset(train_path)),
        HybridGraphLangDataset(load_graph_dataset(val_path)),
        HybridGraphLangDataset(load_graph_dataset(test_path)),
    )


def _subset_targets(subset) -> torch.Tensor:
    """Stack the y tensors of a Subset/Dataset into [N, output_dim] (train-only use)."""
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


def _target_range(train_subset) -> Tuple[float, float]:
    """B2: NRMSE denominator from the TRAIN split only."""
    y = _subset_targets(train_subset).view(-1)
    if y.numel() == 0:
        return (0.0, 0.0)
    return (float(y.min()), float(y.max()))


def _print_target_stats(label: str, train_subset) -> None:
    y = _subset_targets(train_subset).view(-1)
    if y.numel() == 0:
        return
    print(f"[{label}] train target stats: n={y.numel()} mean={y.mean():.3f} std={y.std():.3f} range=[{y.min():.3f}, {y.max():.3f}]")


def _decoder_token_stats(decoder_logits: torch.Tensor, decoder_targets: torch.Tensor, pad_idx: int) -> Tuple[int, int]:
    mask = decoder_targets != pad_idx
    if not mask.any():
        return 0, 0
    predictions = decoder_logits.argmax(dim=-1)
    correct = int(((predictions == decoder_targets) & mask).sum().item())
    total = int(mask.sum().item())
    return correct, total


# --------------------------------------------------------------------------- #
# Eval / train loops
# --------------------------------------------------------------------------- #
def evaluate(
    model,
    loader,
    criterion,
    device,
    target_range: Optional[tuple] = None,
    use_decoder: bool = False,
    training_mode: str = "predictor",
    decoder_vocab: Optional[dict] = None,
    decoder_max_len: int = 96,
    standardizer: Optional[TargetStandardizer] = None,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    decoder_correct_tokens = 0
    decoder_total_tokens = 0
    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    decoder_criterion = nn.CrossEntropyLoss(ignore_index=decoder_vocab["pad_idx"]) if use_decoder and decoder_vocab else None

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if use_decoder and training_mode == "decoder":
                smiles = getattr(batch, "smiles", None)
                if smiles is None:
                    raise ValueError("Decoder evaluation needs batch.smiles")
                decoder_inputs, decoder_targets = encode_smiles_batch(smiles, decoder_vocab, decoder_max_len, device)
                batch_y = getattr(batch, "y", None)
                property_values = (batch_y.float().unsqueeze(-1) if batch_y.dim() == 1 else batch_y.float()) if batch_y is not None else None
                _, decoder_logits = model(batch, decoder_input_ids=decoder_inputs, property_values=property_values)
                loss = decoder_criterion(decoder_logits.view(-1, decoder_logits.size(-1)), decoder_targets.view(-1))
                bc, bt = _decoder_token_stats(decoder_logits, decoder_targets, decoder_vocab["pad_idx"])
                decoder_correct_tokens += bc
                decoder_total_tokens += bt
            else:
                logits = model(batch)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                targets_orig = batch.y.float().view_as(logits)
                targets_std = standardizer.transform(targets_orig) if standardizer is not None else targets_orig
                loss = criterion(logits, targets_std)
                preds_orig = standardizer.inverse_transform(logits) if standardizer is not None else logits
                all_preds.append(preds_orig.detach().cpu())
                all_targets.append(targets_orig.detach().cpu())
            total_loss += loss.item() * batch.num_graphs
            total_items += batch.num_graphs

    metrics = {"loss": total_loss / max(total_items, 1)}
    if use_decoder and training_mode == "decoder":
        metrics["token_accuracy"] = decoder_correct_tokens / max(decoder_total_tokens, 1)
        metrics["perplexity"] = math.exp(min(metrics["loss"], 20))
        return metrics

    preds = torch.cat(all_preds) if all_preds else torch.zeros(0)
    targets = torch.cat(all_targets) if all_targets else torch.zeros(0)
    metrics.update(regression_metrics(preds, targets, target_range))
    return metrics


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    target_range,
    use_decoder: bool,
    training_mode: str,
    decoder_vocab: Optional[dict],
    decoder_loss_weight: float,
    decoder_max_len: int,
    smiles_augment_prob: float = 0.0,
    property_context_dropout_prob: float = 0.0,
    standardizer: Optional[TargetStandardizer] = None,
    grad_clip: float = 0.0,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_items = 0
    decoder_correct_tokens = 0
    decoder_total_tokens = 0
    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    decoder_criterion = nn.CrossEntropyLoss(ignore_index=decoder_vocab["pad_idx"]) if use_decoder and decoder_vocab else None

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        if use_decoder and training_mode in {"decoder", "joint"}:
            smiles = getattr(batch, "smiles", None)
            if smiles is None:
                raise ValueError("Decoder training needs batch.smiles")
            decoder_inputs, decoder_targets = encode_smiles_batch(smiles, decoder_vocab, decoder_max_len, device)
            batch.smiles = _augment_smiles_batch(smiles, smiles_augment_prob)
            batch_y = getattr(batch, "y", None)
            property_values = (batch_y.float().unsqueeze(-1) if batch_y.dim() == 1 else batch_y.float()) if batch_y is not None else None
            context_dropout_mask = None
            if property_context_dropout_prob > 0.0 and property_values is not None:
                context_dropout_mask = torch.rand(batch.num_graphs, device=device) < property_context_dropout_prob

            if training_mode == "decoder":
                _, decoder_logits = model(batch, decoder_input_ids=decoder_inputs, property_values=property_values, decoder_context_dropout_mask=context_dropout_mask)
                loss = decoder_criterion(decoder_logits.view(-1, decoder_logits.size(-1)), decoder_targets.view(-1))
            else:  # joint
                if batch_y is None:
                    raise ValueError("Joint training needs a labeled dataset (batch.y)")
                prop_logits, decoder_logits = model(batch, decoder_input_ids=decoder_inputs, property_values=property_values, decoder_context_dropout_mask=context_dropout_mask)
                targets_orig = batch_y.float().view_as(prop_logits)
                targets_std = standardizer.transform(targets_orig) if standardizer is not None else targets_orig
                prop_loss = criterion(prop_logits, targets_std)
                dec_loss = decoder_criterion(decoder_logits.view(-1, decoder_logits.size(-1)), decoder_targets.view(-1))
                loss = prop_loss + decoder_loss_weight * dec_loss
            bc, bt = _decoder_token_stats(decoder_logits, decoder_targets, decoder_vocab["pad_idx"])
            decoder_correct_tokens += bc
            decoder_total_tokens += bt
        else:
            logits = model(batch)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
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
    if use_decoder and training_mode in {"decoder", "joint"}:
        metrics["token_accuracy"] = decoder_correct_tokens / max(decoder_total_tokens, 1)
    if use_decoder and training_mode == "decoder":
        metrics["perplexity"] = math.exp(min(metrics["loss"], 20))
        return metrics

    if all_preds:
        metrics.update(regression_metrics(torch.cat(all_preds), torch.cat(all_targets), target_range))
    return metrics


def load_graph_pretrained_checkpoint(model, checkpoint_path: str) -> None:
    """Load a training/pretrain_graph.py checkpoint's encoder weights into model.graph_encoder."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("graph_encoder", checkpoint.get("encoder_state_dict", checkpoint.get("model_state_dict", {})))
    if not isinstance(state_dict, dict):
        return

    # A shape mismatch here almost always means --hidden-dim/--num-layers/--graph-backbone
    # don't match the run that produced the checkpoint. Catch it up front with an actionable
    # message instead of letting load_state_dict raise a wall of per-tensor shape errors.
    model_state = model.graph_encoder.state_dict()
    mismatched = [k for k in state_dict if k in model_state and tuple(state_dict[k].shape) != tuple(model_state[k].shape)]
    if mismatched:
        example = mismatched[0]
        raise ValueError(
            f"--graph-pretrained-checkpoint {checkpoint_path!r} was produced with a different graph_encoder "
            f"shape than this run's (--hidden-dim / --num-layers / --graph-backbone must match the pretraining "
            f"run's). E.g. {example}: checkpoint has {tuple(state_dict[example].shape)}, this model has "
            f"{tuple(model_state[example].shape)}. Either drop the mismatched flag(s), or re-run "
            f"training/pretrain_graph.py with this run's --hidden-dim/--num-layers/--graph-backbone to get a "
            f"matching checkpoint. ({len(mismatched)} tensor(s) mismatched in total.)"
        )
    model.graph_encoder.load_state_dict(state_dict, strict=False)


# --------------------------------------------------------------------------- #
# Shared predictor-only loop for the fusion-ablation trainers
# --------------------------------------------------------------------------- #
def run_predictor_ablation_training(model, args, train_set, val_set, test_set, target_range, criterion) -> dict:
    device = args.device
    standardize = getattr(args, "standardize_target", True)
    standardizer = TargetStandardizer(enabled=standardize).fit(_subset_targets(train_set)) if standardize else None
    if standardizer is not None:
        _print_target_stats("ablation", train_set)
    grad_clip = getattr(args, "grad_clip", 1.0)
    warmup_epochs = getattr(args, "warmup_epochs", 5)
    min_lr_ratio = getattr(args, "min_lr_ratio", 0.01)
    lr_schedule = getattr(args, "lr_schedule", "plateau")
    plateau_factor = getattr(args, "plateau_factor", 0.5)
    plateau_patience = getattr(args, "plateau_patience", 5)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(lr_schedule, optimizer, warmup_epochs, args.epochs, min_lr_ratio, plateau_factor, plateau_patience)
    wandb_run = wandb_init(args, config=vars(args))

    train_loader = make_loader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(val_set, args.batch_size, shuffle=False, num_workers=args.num_workers)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, target_range,
            use_decoder=False, training_mode="predictor", decoder_vocab=None,
            decoder_loss_weight=0.0, decoder_max_len=96,
            standardizer=standardizer, grad_clip=grad_clip,
        )
        val_metrics = evaluate(
            model, val_loader, criterion, device, target_range,
            use_decoder=False, training_mode="predictor", decoder_vocab=None,
            standardizer=standardizer,
        )
        step_scheduler(scheduler, val_metrics["loss"])
        print(
            f"Epoch {epoch:03d} | lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"train_loss={train_metrics['loss']:.4f} | val_loss={val_metrics['loss']:.4f} "
            f"| train_rmse={train_metrics.get('rmse', float('nan')):.4f} | val_rmse={val_metrics.get('rmse', float('nan')):.4f}"
        )
        wandb_log(wandb_run, {**{f"train/{k}": v for k, v in train_metrics.items()}, **{f"val/{k}": v for k, v in val_metrics.items()}, "lr": optimizer.param_groups[0]["lr"]}, step=epoch)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            best_state = {"model_state_dict": model.state_dict(), "epoch": epoch, "args": vars(args)}
            if args.checkpoint_path:
                cp = Path(args.checkpoint_path)
                cp.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, cp)
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])

    test_loader = make_loader(test_set, args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_metrics = evaluate(model, test_loader, criterion, device, target_range, use_decoder=False, training_mode="predictor", decoder_vocab=None, standardizer=standardizer)
    print(f"Test loss={test_metrics['loss']:.4f} | Test RMSE={test_metrics.get('rmse', float('nan')):.4f} | Test NRMSE={test_metrics.get('nrmse', float('nan')):.4f}")
    wandb_log(wandb_run, {f"test/{k}": v for k, v in test_metrics.items()})
    wandb_finish(wandb_run)
    return test_metrics


# --------------------------------------------------------------------------- #
# One full train/val/test run for a single seed
# --------------------------------------------------------------------------- #
def run_single(
    args: argparse.Namespace,
    seed: int,
    dataset=None,
    train_set=None,
    val_set=None,
    test_set=None,
) -> Dict[str, float]:
    seed_everything(seed, deterministic=args.deterministic)

    if train_set is None:
        if dataset is None:
            raise ValueError("run_single needs either dataset= (internal split) or train_set/val_set/test_set= (predefined split)")
        all_smiles = [str(getattr(dataset[i], "smiles", "")) for i in range(len(dataset))]
        train_set, val_set, test_set = split_dataset_by_strategy(
            dataset, args.split, args.train_ratio, args.val_ratio, args.test_ratio, seed, smiles_list=all_smiles
        )
        print(f"[seed {seed}] internal split={args.split} -> train={len(train_set)} val={len(val_set)} test={len(test_set)}")
    else:
        print(f"[seed {seed}] predefined split -> train={len(train_set)} val={len(val_set)} test={len(test_set)}")

    decoder_vocab = None
    if args.use_decoder:
        train_smiles = [str(getattr(train_set[idx], "smiles", "")) for idx in range(len(train_set))]
        decoder_vocab = build_smiles_vocab(train_smiles)
        args.decoder_vocab_size = len(decoder_vocab["token_to_id"])
        args.decoder_pad_idx = decoder_vocab["pad_idx"]
        args.decoder_start_idx = decoder_vocab["start_idx"]
        args.decoder_end_idx = decoder_vocab["end_idx"]

    criterion = nn.MSELoss()

    standardizer = TargetStandardizer(enabled=args.standardize_target).fit(_subset_targets(train_set)) if args.standardize_target else None
    target_range = _target_range(train_set)
    _print_target_stats(f"seed {seed}", train_set)

    model = build_model_from_args(args).to(args.device)
    if args.load_checkpoint:
        ckpt = torch.load(args.load_checkpoint, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict"))
        if state_dict is None:
            raise ValueError("Checkpoint does not contain model_state_dict or state_dict")
        model.load_state_dict(state_dict, strict=False)
    if args.graph_pretrained_checkpoint:
        load_graph_pretrained_checkpoint(model, args.graph_pretrained_checkpoint)
    if args.freeze_graph_encoder:
        for parameter in model.graph_encoder.parameters():
            parameter.requires_grad_(False)

    if args.use_decoder and args.training_mode == "decoder":
        sample_graph = train_set[0]
        if not hasattr(sample_graph, "batch") or sample_graph.batch is None:
            sample_graph.batch = torch.zeros(sample_graph.num_nodes, dtype=torch.long)
        sample_graph = sample_graph.to(args.device)
        with torch.no_grad():
            _ = model(
                sample_graph,
                decoder_input_ids=torch.zeros(1, 1, dtype=torch.long, device=args.device),
                property_values=torch.zeros(1, 1, dtype=torch.float, device=args.device),
            )
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("decoder") or name.startswith("decoder_condition_proj"))

    # Linear probing: freeze the (pretrained) graph_encoder for the first --linear-probe-epochs
    # epochs -- so the randomly-initialized head settles down before any gradient reaches the
    # encoder -- then unfreeze it at a reduced LR. Both param groups are created up front (the
    # encoder group's grad stays None, and every torch optimizer skips params with grad=None in
    # .step()) so unfreezing mid-training is just flipping requires_grad, with no optimizer/
    # scheduler rebuild and no disruption to Adam's running moment estimates or the plateau
    # scheduler's patience counter.
    linear_probe_epochs = args.linear_probe_epochs if (args.graph_pretrained_checkpoint and not args.freeze_graph_encoder) else 0
    encoder_params: List[torch.nn.Parameter] = []
    if linear_probe_epochs > 0:
        encoder_params = [p for p in model.graph_encoder.parameters() if p.requires_grad]
        encoder_param_ids = {id(p) for p in encoder_params}
        head_params = [p for p in model.parameters() if p.requires_grad and id(p) not in encoder_param_ids]
        for p in encoder_params:
            p.requires_grad_(False)
        optimizer = torch.optim.AdamW(
            [
                {"params": head_params, "lr": args.lr},
                {"params": encoder_params, "lr": args.lr * args.unfreeze_lr_mult},
            ],
            weight_decay=args.weight_decay,
        )
        print(f"[seed {seed}] linear probe: graph_encoder frozen for {linear_probe_epochs} epoch(s), then unfreezes at lr={args.lr * args.unfreeze_lr_mult:.2e}")
    else:
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(args.lr_schedule, optimizer, args.warmup_epochs, args.epochs, args.min_lr_ratio, args.plateau_factor, args.plateau_patience)

    run_name = f"{args.wandb_run_name}-s{seed}" if args.wandb_run_name else None
    wandb_run = wandb_init(argparse.Namespace(**{**vars(args), "wandb_run_name": run_name}), config={**vars(args), "seed": seed})

    train_loader = make_loader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(val_set, args.batch_size, shuffle=False, num_workers=args.num_workers)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        if linear_probe_epochs > 0 and epoch == linear_probe_epochs + 1:
            for p in encoder_params:
                p.requires_grad_(True)
            print(f"[seed {seed}] linear probe done -> unfreezing graph_encoder at epoch {epoch}")
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, args.device, target_range,
            args.use_decoder, args.training_mode, decoder_vocab, args.decoder_loss_weight, args.decoder_max_len,
            args.smiles_augment_prob, args.property_context_dropout_prob,
            standardizer=standardizer, grad_clip=args.grad_clip,
        )
        val_metrics = evaluate(
            model, val_loader, criterion, args.device, target_range,
            use_decoder=args.use_decoder, training_mode=args.training_mode, decoder_vocab=decoder_vocab,
            decoder_max_len=args.decoder_max_len, standardizer=standardizer,
        )
        step_scheduler(scheduler, val_metrics["loss"])

        lr_now = optimizer.param_groups[0]["lr"]
        if args.use_decoder and args.training_mode == "decoder":
            print(f"Epoch {epoch:03d} | lr={lr_now:.2e} | train_dec_loss={train_metrics['loss']:.4f} | val_dec_loss={val_metrics['loss']:.4f} | val_tok_acc={val_metrics['token_accuracy']:.4f}")
        else:
            print(f"Epoch {epoch:03d} | lr={lr_now:.2e} | train_loss={train_metrics['loss']:.4f} | val_loss={val_metrics['loss']:.4f} | train_rmse={train_metrics.get('rmse', float('nan')):.4f} | val_rmse={val_metrics.get('rmse', float('nan')):.4f} | val_nrmse={val_metrics.get('nrmse', float('nan')):.4f}")

        wandb_log(wandb_run, {**{f"train/{k}": v for k, v in train_metrics.items()}, **{f"val/{k}": v for k, v in val_metrics.items()}, "lr": lr_now}, step=epoch)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            best_state = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "args": vars(args),
                "seed": seed,
                "decoder_vocab": decoder_vocab,
                "target_standardizer": standardizer.state_dict() if standardizer is not None else None,
            }
            cp = Path(args.checkpoint_path) if args.checkpoint_path else Path("checkpoints") / "best.pt"
            cp = cp.with_name(f"{cp.stem}_s{seed}{cp.suffix}") if args.seeds and len(args.seeds) > 1 else cp
            cp.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, cp)
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            print(f"[seed {seed}] early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])

    test_loader = make_loader(test_set, args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_metrics = evaluate(
        model, test_loader, criterion, args.device, target_range,
        use_decoder=args.use_decoder, training_mode=args.training_mode, decoder_vocab=decoder_vocab,
        decoder_max_len=args.decoder_max_len, standardizer=standardizer,
    )
    if args.use_decoder and args.training_mode == "decoder":
        print(f"[seed {seed}] Test dec_loss={test_metrics['loss']:.4f} | token_acc={test_metrics['token_accuracy']:.4f} | ppl={test_metrics['perplexity']:.4f}")
    else:
        print(f"[seed {seed}] Test loss={test_metrics['loss']:.4f} | RMSE={test_metrics.get('rmse', float('nan')):.4f} | NRMSE={test_metrics.get('nrmse', float('nan')):.4f} | MAE={test_metrics.get('mae', float('nan')):.4f}")
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
            raise ValueError("Provide --dataset-dir (or --train-path/--val-path/--test-path) for a predefined split, or --data-path for an internal --split.")
        dataset = HybridGraphLangDataset(load_graph_dataset(args.data_path))
        for seed in seeds:
            per_seed.append(run_single(args, seed, dataset=dataset))

    if len(seeds) > 1:
        print(format_seed_table(aggregate_seed_metrics(per_seed)))


if __name__ == "__main__":
    main()
