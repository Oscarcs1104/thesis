"""Convert a SMILES string into the PyG molecular graph the model actually consumes,
and print its nodes (atoms) and edges (bonds) -- optionally saving a 2D depiction too.

Usage:
  python tools/smiles_to_graph.py --smiles "CCO"
  python tools/smiles_to_graph.py --smiles "CCO" --save-image graph.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rdkit import Chem

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data_pipeline.convert_smiles_to_pyg import smiles_to_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a SMILES into the PyG molecular graph used by the model")
    parser.add_argument("--smiles", required=True, help="Input SMILES string")
    parser.add_argument("--save-image", type=str, default=None, help="Optional path to save a 2D depiction (e.g. graph.png)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = smiles_to_data(args.smiles)
    if data is None:
        raise ValueError(f"Invalid SMILES: {args.smiles}")

    mol = Chem.MolFromSmiles(args.smiles)
    print(f"Input SMILES: {args.smiles}")
    print(f"Canonical SMILES: {data.smiles}")
    print(f"Nodes (atoms): {data.x.size(0)}")
    print(f"Edges (directed, each bond counted twice): {data.edge_index.size(1)}")

    print("\nNode features [atomic_num, degree, formal_charge, num_H, aromatic, chiral_tag, hybridization]:")
    for i, atom in enumerate(mol.GetAtoms()):
        feat_str = ", ".join(f"{v:g}" for v in data.x[i].tolist())
        print(f"  node {i:2d}  {atom.GetSymbol():>2s}  [{feat_str}]")

    print("\nEdges (bonds):")
    seen = set()
    for i in range(data.edge_index.size(1)):
        a, b = int(data.edge_index[0, i]), int(data.edge_index[1, i])
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        bond = mol.GetBondBetweenAtoms(a, b)
        print(f"  {a:2d} -- {b:2d}   (bond order {bond.GetBondTypeAsDouble():g})")

    if args.save_image:
        from rdkit.Chem import Draw

        Draw.MolToFile(mol, args.save_image, size=(500, 500))
        print(f"\nSaved 2D depiction to {args.save_image}")


if __name__ == "__main__":
    main()
