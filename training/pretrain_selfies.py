"""Masked-LM pretraining over SELFIES tokens.

Independent from train.py's HuggingFace language branch: this produces a small
custom encoder pretrained on molecule tokens, saved with its own vocabulary.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.smiles_decoder import build_vocab, tokenize_molecule
from training.wandb_utils import add_wandb_args, wandb_finish, wandb_init, wandb_log


def _build_mlm_vocab(texts) -> dict:
    vocab = build_vocab(texts)
    mask_idx = len(vocab["token_to_id"])
    vocab["token_to_id"]["<MASK>"] = mask_idx
    vocab["id_to_token"][mask_idx] = "<MASK>"
    vocab["mask_idx"] = mask_idx
    return vocab


def _encode_fixed(text: str, vocab: dict, max_len: int) -> list[int]:
    token_to_id = vocab["token_to_id"]
    unk_idx = vocab["unk_idx"]
    pad_idx = vocab["pad_idx"]
    ids = [token_to_id.get(token, unk_idx) for token in tokenize_molecule(text)]
    ids = ids[:max_len]
    return ids + [pad_idx] * (max_len - len(ids))


class SelfiesTokenDataset(Dataset):
    """Tokenizes+pads one text per item; the SELFIES tokenization work happens here
    so DataLoader workers can parallelize it across CPUs."""

    def __init__(self, texts, vocab: dict, max_len: int) -> None:
        self.texts = texts
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(_encode_fixed(self.texts[idx], self.vocab, self.max_len), dtype=torch.long)


class MaskedLanguagePretrainer(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        return self.proj(h)


def train_selfies(
    smiles_file: str,
    out: str,
    epochs: int = 3,
    batch_size: int = 64,
    device: str = "cpu",
    hidden_dim: int = 256,
    max_len: int = 128,
    max_vocab_samples: int = 20000,
    num_workers: int = 0,
    wandb_run=None,
) -> None:
    path = Path(smiles_file)
    with path.open("r", encoding="utf-8") as handle:
        texts = [line.strip() for line in handle if line.strip()]

    vocab = _build_mlm_vocab(texts[:max_vocab_samples])
    pad_idx = vocab["pad_idx"]
    mask_idx = vocab["mask_idx"]

    model = MaskedLanguagePretrainer(len(vocab["token_to_id"]), hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)

    loader = DataLoader(
        SelfiesTokenDataset(texts, vocab, max_len),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    for epoch in range(1, epochs + 1):
        start = time.time()
        total_loss = 0.0
        num_batches = 0
        for ids in loader:
            ids = ids.to(device)
            mask = (torch.rand_like(ids.float()) < 0.15) & (ids != pad_idx)
            if not mask.any():
                continue
            masked_ids = ids.clone()
            masked_ids[mask] = mask_idx

            logits = model(masked_ids)
            loss = F.cross_entropy(logits[mask], ids[mask])

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            num_batches += 1

        elapsed = int(time.time() - start)
        epoch_loss = total_loss / max(1, num_batches)
        print(f"SELFIES epoch {epoch}/{epochs} | loss={epoch_loss:.4f} | elapsed {elapsed}s")
        wandb_log(wandb_run, {"train/loss": epoch_loss}, step=epoch)

    torch.save(
        {
            "stage": "selfies_lm",
            "model_state_dict": model.state_dict(),
            "vocab": vocab,
            "hidden_dim": hidden_dim,
            "max_len": max_len,
        },
        out,
    )
    print(f"Saved SELFIES pretrain to {out}")
    wandb_finish(wandb_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Masked-LM pretraining over SELFIES tokens")
    parser.add_argument("--smiles-file", type=str, required=True, help="Text file with one SMILES per line")
    parser.add_argument("--out", type=str, default="selfies_pretrain.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0, help="Parallel CPU processes for tokenization (e.g. 4 if your machine has 4 CPUs)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--max-vocab-samples", type=int, default=20000, help="Cap on lines used to build the token vocabulary")
    add_wandb_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wandb_run = wandb_init(args, config=vars(args))
    train_selfies(
        args.smiles_file,
        args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        hidden_dim=args.hidden_dim,
        max_len=args.max_len,
        max_vocab_samples=args.max_vocab_samples,
        num_workers=args.num_workers,
        wandb_run=wandb_run,
    )


if __name__ == "__main__":
    main()
