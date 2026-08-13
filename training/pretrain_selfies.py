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

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.smiles_decoder import build_vocab, tokenize_molecule


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
) -> None:
    path = Path(smiles_file)
    with path.open("r", encoding="utf-8") as handle:
        texts = [line.strip() for line in handle if line.strip()]

    vocab = _build_mlm_vocab(texts[:max_vocab_samples])
    pad_idx = vocab["pad_idx"]
    mask_idx = vocab["mask_idx"]

    model = MaskedLanguagePretrainer(len(vocab["token_to_id"]), hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)

    for epoch in range(1, epochs + 1):
        start = time.time()
        total_loss = 0.0
        num_batches = 0
        for batch_start in range(0, len(texts), batch_size):
            batch_texts = texts[batch_start:batch_start + batch_size]
            ids = torch.tensor([_encode_fixed(text, vocab, max_len) for text in batch_texts], dtype=torch.long, device=device)
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
        print(f"SELFIES epoch {epoch}/{epochs} | loss={total_loss / max(1, num_batches):.4f} | elapsed {elapsed}s")

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Masked-LM pretraining over SELFIES tokens")
    parser.add_argument("--smiles-file", type=str, required=True, help="Text file with one SMILES per line")
    parser.add_argument("--out", type=str, default="selfies_pretrain.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--max-vocab-samples", type=int, default=20000, help="Cap on lines used to build the token vocabulary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_selfies(
        args.smiles_file,
        args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        hidden_dim=args.hidden_dim,
        max_len=args.max_len,
        max_vocab_samples=args.max_vocab_samples,
    )


if __name__ == "__main__":
    main()
