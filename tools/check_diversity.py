"""Check 2D diversity of a molecule dataset: internal Tanimoto diversity + Shannon
entropy over Murcko scaffolds.

Works on anything data_pipeline.data.load_graph_dataset accepts (a CSV, a cached
.graphs.pt, or a folder like data/zinc), so it can check ZINC, esol/freesolv/lipo,
or any other dataset in the same format.

Usage:
  python tools/check_diversity.py --data-path data/zinc15_250K.csv --sample-size 2000
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data_pipeline.data import load_graph_dataset
from tools.mol_metrics import mean_pairwise_tanimoto, mols_from_smiles, morgan_fingerprints, murcko_scaffold_smiles, shannon_entropy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check 2D diversity of a molecule dataset")
    parser.add_argument("--data-path", type=str, required=True, help="Same format as train.py --data-path")
    parser.add_argument("--sample-size", type=int, default=2000, help="Cap on molecules used for the O(n^2) pairwise Tanimoto calculation")
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graphs = load_graph_dataset(args.data_path)
    smiles_list = [s for s in (getattr(g, "smiles", None) for g in graphs) if s]
    print(f"Loaded {len(smiles_list)} molecules with SMILES from {args.data_path}")

    mols, invalid = mols_from_smiles(smiles_list)
    if invalid:
        print(f"Skipped {invalid} SMILES that failed to parse")

    random.seed(args.seed)
    sample = mols if len(mols) <= args.sample_size else random.sample(mols, args.sample_size)
    print(f"Pairwise Tanimoto computed on a sample of {len(sample)} (of {len(mols)} total)")

    fps = morgan_fingerprints(sample)
    mean_sim = mean_pairwise_tanimoto(fps)
    if mean_sim is None:
        print("Not enough molecules for pairwise similarity")
    else:
        print(f"Mean pairwise Tanimoto similarity: {mean_sim:.4f}")
        print(f"Mean pairwise Tanimoto distance (diversity): {1 - mean_sim:.4f}")

    scaffolds = [murcko_scaffold_smiles(mol) for mol in mols]
    unique_scaffolds = len(set(scaffolds))
    entropy = shannon_entropy(scaffolds)
    print(f"Unique Murcko scaffolds: {unique_scaffolds} / {len(mols)} molecules ({unique_scaffolds / max(len(mols), 1):.2%})")
    print(f"Shannon entropy over scaffolds: {entropy:.4f}")


if __name__ == "__main__":
    main()
