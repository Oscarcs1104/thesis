"""Cap how many molecules per Murcko scaffold survive in a dataset, to raise the
Shannon entropy over scaffolds (reduce domination by a handful of common scaffolds).

Groups molecules by scaffold, and for any scaffold with more than --max-per-scaffold
molecules, randomly keeps only that many -- the rest are dropped. Prints before/after
scaffold count and entropy so you can see the effect directly.

Usage:
  python data_pipeline/scaffold_balanced_sample.py --input data/zinc15_1M.csv --output data/zinc15_1M_balanced.csv --max-per-scaffold 8
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tools.mol_metrics import shannon_entropy

RDLogger.DisableLog("rdApp.*")


def _scaffold_for(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold) if scaffold is not None else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cap molecules per Murcko scaffold to raise scaffold entropy")
    parser.add_argument("--input", type=str, required=True, help="Input CSV with a smiles column")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument("--max-per-scaffold", type=int, default=8, help="Max molecules kept per unique scaffold")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel processes for scaffold computation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    with input_path.open("r", encoding="utf-8") as handle:
        smiles_list = [row["smiles"] for row in csv.DictReader(handle) if row.get("smiles")]
    print(f"Loaded {len(smiles_list)} molecules from {input_path}")

    print("Computing Murcko scaffolds...")
    scaffolds = Parallel(n_jobs=args.n_jobs)(delayed(_scaffold_for)(smi) for smi in smiles_list)

    groups: Dict[str, List[int]] = defaultdict(list)
    invalid = 0
    for idx, scaffold in enumerate(scaffolds):
        if scaffold is None:
            invalid += 1
            continue
        groups[scaffold].append(idx)
    if invalid:
        print(f"Skipped {invalid} molecules that failed to parse")

    valid_scaffolds = [s for s in scaffolds if s is not None]
    before_entropy = shannon_entropy(valid_scaffolds)
    print(f"Before: {len(valid_scaffolds)} molecules, {len(groups)} unique scaffolds, entropy={before_entropy:.4f}")

    random.seed(args.seed)
    kept_indices: List[int] = []
    capped_groups = 0
    for scaffold, indices in groups.items():
        if len(indices) > args.max_per_scaffold:
            capped_groups += 1
            indices = random.sample(indices, args.max_per_scaffold)
        kept_indices.extend(indices)
    kept_indices.sort()

    kept_scaffolds = [scaffolds[i] for i in kept_indices]
    after_entropy = shannon_entropy(kept_scaffolds)
    print(f"Capped {capped_groups} scaffolds down to {args.max_per_scaffold} molecules each")
    print(f"After:  {len(kept_indices)} molecules, {len(set(kept_scaffolds))} unique scaffolds, entropy={after_entropy:.4f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["smiles"])
        for idx in kept_indices:
            writer.writerow([smiles_list[idx]])
    print(f"Wrote {len(kept_indices)} molecules to {output_path}")


if __name__ == "__main__":
    # Re-import as a real module (not __main__) so joblib/loky can pickle the
    # per-molecule worker function by reference instead of by value.
    import data_pipeline.scaffold_balanced_sample as _self

    _self.main()
