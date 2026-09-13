"""Inspect the model's internal representations: `fused_feat` (the shared graph+
language vector fed to the property head) vs. `decoder_latent` (the actual
condition token that feeds the SmilesDecoder, built from fused_feat + the target
property via `decoder_condition_proj`). These are two different tensors -- this
tool lets you look at both.

Two modes:
  --smiles      print summary stats for one molecule's fused_feat and decoder_latent.
  --data-path   embed every molecule in a dataset, project both fused_feat and
                decoder_latent to 2D with PCA, and save a scatter plot of each
                colored by the real property value -- to see whether either space
                actually organizes itself by property.

See COMMANDS.md for usage.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data_pipeline.convert_smiles_to_pyg import smiles_to_data
from data_pipeline.data import load_graph_dataset
from thesis_model.generation.demo_generate_property import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the fused representation and decoder condition latent")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--smiles", type=str, default=None, help="Inspect a single molecule")
    parser.add_argument("--data-path", type=str, default=None, help="Inspect a whole dataset and plot 2D PCA scatters")
    parser.add_argument("--property-values", type=float, nargs="+", default=None, help="Property to condition on in single-molecule mode; default: the model's own prediction")
    parser.add_argument("--max-molecules", type=int, default=300, help="Cap on molecules embedded in dataset mode (random sample if the dataset is bigger)")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--out-dir", type=str, default="plots", help="Where to save the PCA scatter plots (dataset mode)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _summarize(name: str, vec: torch.Tensor) -> None:
    v = vec.detach().cpu().float().view(-1)
    print(f"{name}: dim={v.numel()}  norm={v.norm().item():.4f}  mean={v.mean().item():.4f}  std={v.std().item():.4f}  min={v.min().item():.4f}  max={v.max().item():.4f}")
    print(f"  first 10 components: {[round(x, 4) for x in v[:10].tolist()]}")


def _pca_2d(x: np.ndarray) -> np.ndarray:
    """Project rows of x to 2D via PCA (mean-centered SVD) -- no sklearn dependency."""
    centered = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def inspect_single(args: argparse.Namespace, model, device: torch.device) -> None:
    data = smiles_to_data(args.smiles)
    if data is None:
        raise ValueError("Invalid SMILES")
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    data = data.to(device)

    with torch.no_grad():
        fused_feat = model.encode(data)
        pred = model.head(fused_feat)
        property_values = (
            torch.tensor([args.property_values], dtype=torch.float, device=device)
            if args.property_values is not None
            else pred
        )
        decoder_latent = model._build_decoder_latent(fused_feat, property_values=property_values)

    print(f"SMILES: {args.smiles}")
    print(f"Predicted property: {pred.squeeze(-1).item():.4f}")
    print(f"Property used to condition the decoder: {property_values.squeeze(-1).item():.4f}")
    print()
    _summarize("fused_feat (predictor input -- what the model uses to predict)", fused_feat)
    print()
    _summarize("decoder_latent (condition token -- what actually enters the SmilesDecoder)", decoder_latent)


def inspect_dataset(args: argparse.Namespace, model, device: torch.device) -> None:
    graphs = load_graph_dataset(args.data_path)
    random.seed(args.seed)
    if len(graphs) > args.max_molecules:
        graphs = random.sample(graphs, args.max_molecules)

    fused_list, latent_list, y_list = [], [], []
    with torch.no_grad():
        for g in graphs:
            g = g.clone()
            g.batch = torch.zeros(g.num_nodes, dtype=torch.long)
            g = g.to(device)
            y = getattr(g, "y", None)
            y_val = float(y.view(-1)[0]) if y is not None else float("nan")

            fused = model.encode(g)
            property_values = torch.tensor([[y_val]], dtype=torch.float, device=device) if y is not None else None
            latent = model._build_decoder_latent(fused, property_values=property_values)

            fused_list.append(fused.squeeze(0).cpu().numpy())
            latent_list.append(latent.squeeze(0).cpu().numpy())
            y_list.append(y_val)

    x_fused = np.stack(fused_list)
    x_latent = np.stack(latent_list)
    y = np.array(y_list)
    print(f"Embedded {len(fused_list)} molecules from {args.data_path}")
    print(f"fused_feat dim={x_fused.shape[1]}  decoder_latent dim={x_latent.shape[1]}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, x in (("fused_feat", x_fused), ("decoder_latent", x_latent)):
        coords = _pca_2d(x)
        plt.figure(figsize=(7, 6))
        sc = plt.scatter(coords[:, 0], coords[:, 1], c=y, cmap="viridis", s=14, alpha=0.85)
        plt.colorbar(sc, label="real property (y)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"PCA of {name} -- {args.data_path}")
        plt.tight_layout()
        out_path = out_dir / f"latent_pca_{name}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved {out_path}")


def main() -> None:
    args = parse_args()
    if not args.smiles and not args.data_path:
        raise ValueError("Pass --smiles (single molecule) or --data-path (whole dataset)")

    device = torch.device(args.device)
    model, _ = build_model(args.checkpoint_path, str(device))

    if args.smiles:
        inspect_single(args, model, device)
    if args.data_path:
        inspect_dataset(args, model, device)


if __name__ == "__main__":
    main()
