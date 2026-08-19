"""Check 2D diversity of a molecule dataset: Shannon entropy over Murcko scaffolds.

Works on anything data_pipeline.data.load_graph_dataset accepts (a CSV, a cached
.graphs.pt, or a folder like data/zinc), so it can check ZINC, esol/freesolv/lipo,
or any other dataset in the same format.

Pairwise Tanimoto diversity is not computed here (it's an O(n^2) estimate that gets
noisy once sampled down for large datasets) -- that metric lives in
demo_generate_property.py, where it runs on the small set of generated candidates.

Usage:
  python tools/check_diversity.py --data-path data/zinc15_250K.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data_pipeline.data import load_graph_dataset
from tools.mol_metrics import mols_from_smiles, murcko_scaffold_smiles, shannon_entropy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check 2D diversity of a molecule dataset")
    parser.add_argument("--data-path", type=str, required=True, help="Same format as train.py --data-path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graphs = load_graph_dataset(args.data_path)
    smiles_list = [s for s in (getattr(g, "smiles", None) for g in graphs) if s]
    print(f"Loaded {len(smiles_list)} molecules with SMILES from {args.data_path}")

    mols, invalid = mols_from_smiles(smiles_list)
    if invalid:
        print(f"Skipped {invalid} SMILES that failed to parse")

    scaffolds = [murcko_scaffold_smiles(mol) for mol in mols]
    unique_scaffolds = len(set(scaffolds))
    entropy = shannon_entropy(scaffolds)
    print(f"Unique Murcko scaffolds: {unique_scaffolds} / {len(mols)} molecules ({unique_scaffolds / max(len(mols), 1):.2%})")
    print(f"Shannon entropy over scaffolds: {entropy:.4f}")


if __name__ == "__main__":
    main()
