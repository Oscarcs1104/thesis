"""Train the standalone conditional SELFIES generator (plan Days 8 + 10).

  --mode pretrain   causal LM over ZINC-250k SELFIES, unconditional (all samples
                    use the null property bin). This is the "6-layer decoder-only
                    on SELFIES / ZINC" step.
  --mode finetune   conditional: load the pretrained generator, bin each molecule
                    by its (pseudo-labelled) property value, prepend the bin token,
                    keep training the same next-token objective. --cond-dropout
                    randomly drops the condition to the null bin (CFG-style).

Checkpoints carry {generator_state, vocab, binner, args} so tools/eval_generation.py
can reload without re-specifying anything.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from model.conditional_generator import ConditionalSmilesGenerator, PropertyBinner
from model.smiles_decoder import build_vocab, encode_batch
from training.repro import build_scheduler, seed_everything, step_scheduler
from training.wandb_utils import add_wandb_args, wandb_finish, wandb_init, wandb_log


# --------------------------------------------------------------------------- #
def _read_smiles_column(path: str, limit: Optional[int] = None) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        col = next((c for c in fields if c and c.lower() in {"smiles", "smile", "canonical_smiles"}), fields[0])
        for row in reader:
            s = (row.get(col) or "").strip()
            if s:
                out.append(s)
            if limit and len(out) >= limit:
                break
    return out


def _read_smiles_and_values(path: str, value_col: Optional[str], limit: Optional[int] = None):
    smiles, values = [], []
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        smi_col = next((c for c in fields if c and c.lower() in {"smiles", "smile", "canonical_smiles"}), fields[0])
        val_col = value_col or next((c for c in fields if c and c != smi_col), None)
        for row in reader:
            s = (row.get(smi_col) or "").strip()
            try:
                v = float(row.get(val_col))
            except (TypeError, ValueError):
                continue
            if s:
                smiles.append(s)
                values.append(v)
            if limit and len(smiles) >= limit:
                break
    return smiles, values


class GenDataset(Dataset):
    def __init__(self, smiles: List[str], vocab: dict, max_len: int, bins: Optional[List[int]] = None, null_bin: int = 0) -> None:
        self.smiles = smiles
        self.vocab = vocab
        self.max_len = max_len
        self.bins = bins
        self.null_bin = null_bin

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int):
        bin_idx = self.null_bin if self.bins is None else self.bins[idx]
        return self.smiles[idx], int(bin_idx)


def _make_collate(vocab: dict, max_len: int, cond_dropout: float, null_bin: int):
    def collate(items):
        texts = [t for t, _ in items]
        bins = torch.tensor([b for _, b in items], dtype=torch.long)
        if cond_dropout > 0:
            drop = torch.rand(len(items)) < cond_dropout
            bins = torch.where(drop, torch.full_like(bins, null_bin), bins)
        inputs, targets = encode_batch(texts, vocab, max_len, torch.device("cpu"))
        return bins, inputs, targets

    return collate


def _token_accuracy(logits: torch.Tensor, targets: torch.Tensor, pad_idx: int) -> tuple:
    mask = targets != pad_idx
    if not mask.any():
        return 0, 0
    correct = int(((logits.argmax(-1) == targets) & mask).sum().item())
    return correct, int(mask.sum().item())


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["pretrain", "finetune"], required=True)
    p.add_argument("--smiles-csv", type=str, required=True,
                   help="pretrain: ZINC SMILES CSV. finetune: pseudo-labelled CSV (smiles + predicted value).")
    p.add_argument("--value-col", type=str, default=None, help="finetune: column with the (pseudo) property value; default = first non-SMILES column")
    p.add_argument("--property-ref-csv", type=str, default=None,
                   help="finetune: CSV whose target column defines the quantile bin edges (use the dataset's csv/train.csv so bins match evaluation)")
    p.add_argument("--property-name", type=str, default="property")
    p.add_argument("--num-bins", type=int, default=10)
    p.add_argument("--cond-dropout", type=float, default=0.1, help="finetune: prob. of replacing the condition with the null bin (CFG)")
    p.add_argument("--load-generator", type=str, default=None, help="finetune: pretrained generator checkpoint from --mode pretrain")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--decoder-layers", type=int, default=6)
    p.add_argument("--decoder-heads", type=int, default=8)
    p.add_argument("--max-len", type=int, default=96)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-frac", type=float, default=0.02)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="Cap the number of molecules (debugging)")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    add_wandb_args(p)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)

    # ----- data + vocab + binner -----
    binner: Optional[PropertyBinner] = None
    bins_per_sample: Optional[List[int]] = None

    if args.mode == "pretrain":
        smiles = _read_smiles_column(args.smiles_csv, args.limit)
        pretrained = None
        vocab = build_vocab(smiles)
        num_bins = args.num_bins  # embedding is sized for later fine-tuning
        null_bin = num_bins
    else:
        if not args.load_generator:
            raise ValueError("--mode finetune needs --load-generator (a --mode pretrain checkpoint)")
        pretrained = torch.load(args.load_generator, map_location="cpu", weights_only=False)
        vocab = pretrained["vocab"]
        num_bins = pretrained["args"]["num_bins"]
        null_bin = num_bins
        smiles, values = _read_smiles_and_values(args.smiles_csv, args.value_col, args.limit)
        ref_values = values
        if args.property_ref_csv:
            _, ref_values = _read_smiles_and_values(args.property_ref_csv, None)
        binner = PropertyBinner.fit(ref_values, num_bins=num_bins, name=args.property_name)
        bins_per_sample = binner.to_bins(values).tolist()
        print(f"[finetune] {len(smiles)} molecules | bins from {args.property_ref_csv or 'pseudo-labels'} | edges={['%.3f' % e for e in binner.edges]}")

    # ----- train / val split -----
    idx = list(range(len(smiles)))
    random.Random(args.seed).shuffle(idx)
    n_val = max(1, int(len(idx) * args.val_frac))
    val_idx, train_idx = set(idx[:n_val]), idx[n_val:]

    def subset(indices):
        s = [smiles[i] for i in indices]
        b = [bins_per_sample[i] for i in indices] if bins_per_sample is not None else None
        return s, b

    tr_s, tr_b = subset(train_idx)
    va_s, va_b = subset(sorted(val_idx))

    collate = _make_collate(vocab, args.max_len, args.cond_dropout if args.mode == "finetune" else 0.0, null_bin)
    train_loader = DataLoader(GenDataset(tr_s, vocab, args.max_len, tr_b, null_bin), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, collate_fn=collate, persistent_workers=args.num_workers > 0)
    val_collate = _make_collate(vocab, args.max_len, 0.0, null_bin)
    val_loader = DataLoader(GenDataset(va_s, vocab, args.max_len, va_b, null_bin), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, collate_fn=val_collate, persistent_workers=args.num_workers > 0)

    # ----- model -----
    model = ConditionalSmilesGenerator(
        hidden_dim=args.hidden_dim,
        vocab_size=len(vocab["token_to_id"]),
        pad_idx=vocab["pad_idx"], start_idx=vocab["start_idx"], end_idx=vocab["end_idx"],
        num_property_bins=num_bins,
        decoder_layers=args.decoder_layers, decoder_heads=args.decoder_heads,
        max_len=max(args.max_len, 128), dropout=args.dropout,
    ).to(device)
    if args.mode == "finetune":
        model.load_state_dict(pretrained["generator_state"], strict=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler("plateau", optimizer, args.warmup_epochs, args.epochs, 0.01, 0.5, 3)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["pad_idx"])
    wandb_run = wandb_init(args, config=vars(args))

    def run_epoch(loader, train: bool):
        model.train(train)
        total_loss, n_batches, correct, total = 0.0, 0, 0, 0
        for bins, inputs, targets in loader:
            bins, inputs, targets = bins.to(device), inputs.to(device), targets.to(device)
            with torch.set_grad_enabled(train):
                logits = model(bins, inputs)
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            c, t = _token_accuracy(logits, targets, vocab["pad_idx"])
            correct += c
            total += t
        avg = total_loss / max(n_batches, 1)
        return {"loss": avg, "ppl": math.exp(min(avg, 20)), "token_acc": correct / max(total, 1)}

    best_val, best_state = float("inf"), None
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(train_loader, True)
        va = run_epoch(val_loader, False)
        step_scheduler(scheduler, va["loss"])
        print(f"Epoch {epoch:03d} | lr={optimizer.param_groups[0]['lr']:.2e} | "
              f"train loss={tr['loss']:.4f} ppl={tr['ppl']:.2f} acc={tr['token_acc']:.3f} | "
              f"val loss={va['loss']:.4f} ppl={va['ppl']:.2f} acc={va['token_acc']:.3f}")
        wandb_log(wandb_run, {**{f"train/{k}": v for k, v in tr.items()}, **{f"val/{k}": v for k, v in va.items()},
                              "lr": optimizer.param_groups[0]["lr"]}, step=epoch)
        if va["loss"] < best_val:
            best_val = va["loss"]
            best_state = {
                "generator_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "vocab": vocab,
                "binner": binner.state_dict() if binner is not None else None,
                "property_name": args.property_name,
                "args": vars(args),
                "epoch": epoch,
            }
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, args.out)

    print(f"Saved best generator (val loss {best_val:.4f}) to {args.out}")
    wandb_finish(wandb_run)


if __name__ == "__main__":
    main()
