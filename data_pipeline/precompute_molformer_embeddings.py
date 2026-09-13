"""Precompute frozen IBM MoLFormer-XL-both-10pct embeddings for a dataset's
molecules, replicating how MoLA (C:\\CVAIL\\Thesis\\MoLA) used MoLFormer: masked
mean-pooling over the frozen backbone's last_hidden_state, computed once and
cached to disk -- never re-run inside the training loop.

See COMMANDS.md for usage.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data_pipeline.data import load_graph_dataset

_MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute frozen IBM MoLFormer embeddings (MoLA-style)")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = torch.sum(masked, dim=1)
    denom = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / denom


def main() -> None:
    args = parse_args()
    from transformers import AutoModel, AutoTokenizer

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(_MODEL_NAME, trust_remote_code=True).to(device).eval()

    graphs = load_graph_dataset(args.data_path)
    smiles_list = [s for s in (getattr(g, "smiles", None) for g in graphs) if s]
    unique_smiles = sorted(set(smiles_list))
    print(f"Computing frozen MoLFormer embeddings for {len(unique_smiles)} unique molecules...")

    embeddings: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for start in range(0, len(unique_smiles), args.batch_size):
            batch_smiles = unique_smiles[start : start + args.batch_size]
            encoded = tokenizer(batch_smiles, padding=True, truncation=True, max_length=202, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            pooled = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"]).cpu()
            for smi, emb in zip(batch_smiles, pooled):
                embeddings[smi] = emb
            print(f"  {min(start + args.batch_size, len(unique_smiles))}/{len(unique_smiles)}", end="\r")

    print()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, out_path)
    emb_dim = next(iter(embeddings.values())).numel()
    print(f"Saved {len(embeddings)} embeddings ({emb_dim}-d) to {out_path}")


if __name__ == "__main__":
    main()
