"""Convert a CSV of SMILES to a list of PyG Data objects using RDKit.

Usage examples:
python convert_smiles_to_pyg.py --csv datasets/lipo/raw/lipo.csv --smiles-col smiles --target-col y --out datasets/lipo/graphs_from_smiles.pt
python convert_smiles_to_pyg.py --dataset-dir datasets/lipo --out datasets/lipo/graphs_from_smiles.pt
"""
from pathlib import Path
import argparse
import csv
import gzip
import torch
from typing import Optional

try:
    from rdkit import Chem
except Exception as e:
    raise RuntimeError("RDKit is required for converting SMILES. Install it in your environment.")

from torch_geometric.data import Data
import numpy as np


def atom_features(atom: Chem.Atom) -> np.ndarray:
    # Basic atom-level features: atomic number, degree, formal charge, num implicit Hs, aromatic, chiral tag, hybridization
    an = atom.GetAtomicNum()
    deg = atom.GetDegree()
    chg = atom.GetFormalCharge()
    nh = atom.GetTotalNumHs()
    aromatic = 1 if atom.GetIsAromatic() else 0
    chiral = int(atom.GetChiralTag())
    hyb = int(atom.GetHybridization())
    return np.array([an, deg, chg, nh, aromatic, chiral, hyb], dtype=np.float32)


def smiles_to_data(smiles: str, target: Optional[float] = None) -> Optional[Data]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # node features
    feats = [atom_features(a) for a in mol.GetAtoms()]
    x = torch.tensor(np.vstack(feats), dtype=torch.float)

    # edges
    edges = []
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])
    if len(edges) > 0:
        edge_arr = np.array(edges, dtype=np.int64).T  # shape [2, E]
        edge_index = torch.tensor(edge_arr, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    if target is not None:
        try:
            data.y = torch.tensor([float(target)], dtype=torch.float)
        except Exception:
            data.y = torch.tensor([0.0], dtype=torch.float)
    # Store the canonical form so every downstream consumer (language branch,
    # decoder target) sees a consistent string for the same molecule.
    data.smiles = Chem.MolToSmiles(mol)
    return data


def randomize_smiles(smiles: str) -> str:
    """Return a randomized (non-canonical) SMILES for the same molecule.

    Falls back to the input unchanged if it can't be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol, canonical=False, doRandom=True)


def find_csv_in_dataset_dir(dataset_dir: Path) -> Optional[Path]:
    # look for raw/*.csv or train_*.csv
    raw = dataset_dir / 'raw'
    if raw.exists():
        for f in raw.glob('*.csv'):
            return f
    for f in dataset_dir.glob('*.csv'):
        return f
    # molprop-style
    for f in dataset_dir.parent.glob('MolPROP/data/*.csv'):
        return f
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default=None, help='Path to CSV file with SMILES')
    parser.add_argument('--dataset-dir', type=str, default=None, help='Dataset folder to search for CSVs')
    parser.add_argument('--smiles-col', type=str, default='smiles')
    parser.add_argument('--target-col', type=str, default=None)
    parser.add_argument('--out', type=str, required=True, help='Output .pt file to save list of Data')
    args = parser.parse_args()

    csv_path = None
    if args.csv:
        csv_path = Path(args.csv)
    elif args.dataset_dir:
        csv_path = find_csv_in_dataset_dir(Path(args.dataset_dir))
    if csv_path is None or not csv_path.exists():
        raise FileNotFoundError('CSV not found. Provide --csv or --dataset-dir pointing to dataset with CSV')

    data_list = []
    # support gzipped csv
    open_f = gzip.open if str(csv_path).endswith('.gz') else open
    with open_f(csv_path, 'rt', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        # if target_col not given, try to pick a numeric column other than smiles
        target_col = args.target_col
        if target_col is None:
            # detect smiles column if not default
            smiles_candidates = [c for c in reader.fieldnames if c.lower() in ('smiles', 'smiles', 'smiles_smiles', 'smile', 'canonical_smiles')]
            if smiles_candidates:
                args.smiles_col = smiles_candidates[0]

            for k in reader.fieldnames:
                if k.lower() == args.smiles_col.lower():
                    continue
                # pick first non-empty column as target
                target_col = k
                break
        for row in reader:
            smi = row.get(args.smiles_col) or row.get(args.smiles_col.lower())
            if smi is None:
                continue
            target = None
            if target_col and target_col in row:
                target = row[target_col]
            data = smiles_to_data(smi, target)
            if data is not None:
                data_list.append(data)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data_list, outp)
    print(f"Saved {len(data_list)} graphs to {outp}")

if __name__ == '__main__':
    main()
