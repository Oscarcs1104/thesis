"""Download the raw MoleculeNet regression CSVs and write a frozen scaffold split.

No deepchem / tensorflow needed: the three CSVs live in DeepChem's public S3
bucket and are already in raw target units. Scaffold splitting uses
`data_pipeline/splitters.py` (deterministic Bemis-Murcko, largest scaffold groups
to train first -- the same policy as DeepChem's ScaffoldSplitter; not guaranteed
byte-identical to `dc.molnet.load_*(splitter="scaffold")`, but the standard
scaffold-split protocol).

    python data_pipeline/download_molnet.py --output-dir data/deepchem_molnet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from data_pipeline.features import canonicalize_smiles
from data_pipeline.splitters import split_dataset

RDLogger.DisableLog("rdApp.*")

_BUCKET = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/"

# name -> (remote file, smiles column, target column)
DATASETS = {
    "delaney": ("delaney-processed.csv", "smiles", "measured log solubility in mols per litre"),
    "freesolv": ("SAMPL.csv", "smiles", "expt"),
    "lipo": ("Lipophilicity.csv", "smiles", "exp"),
}


def _inchikey(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def download_one(name: str, parent_dir: Path, split: str = "scaffold",
                 fracs=(0.8, 0.1, 0.1), seed: int = 2025, force: bool = False,
                 n_mad: float = 5.0, drop_outliers: bool = False) -> None:
    remote_file, smi_col, tgt_col = DATASETS[name]
    raw_dir = parent_dir / name / "raw"
    csv_dir = parent_dir / name / "csv"
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / remote_file
    if raw_path.exists() and not force:
        df = pd.read_csv(raw_path)
    else:
        print(f"[{name}] downloading {_BUCKET + remote_file}")
        df = pd.read_csv(_BUCKET + remote_file)
        df.to_csv(raw_path, index=False)

    df = df[[smi_col, tgt_col]].rename(columns={smi_col: "smiles", tgt_col: "y"})
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["smiles", "y"])

    # Curation before anything else: a SMILES with a dot is more than one species,
    # usually a parent plus a counterion, and the measured property belongs to the
    # parent. See data_pipeline/curation.py for why outliers are treated differently.
    from data_pipeline.curation import curation_summary, find_outliers, strip_fragments

    cleaned, frag_report = strip_fragments(df["smiles"].astype(str).tolist())
    df["smiles"] = cleaned
    df = df.dropna(subset=["smiles"])

    outliers = find_outliers(df["y"].to_numpy(), n_mad=n_mad)
    print(curation_summary(frag_report, outliers, name))
    if drop_outliers and outliers["n_outliers"]:
        keep = np.ones(len(df), dtype=bool)
        keep[outliers["indices"]] = False
        df = df[keep]
        print(f"  [{name}] --drop-outliers: removed {outliers['n_outliers']} rows. "
              f"These numbers are NO LONGER comparable to published MoleculeNet results.")

    df["smiles"] = df["smiles"].map(canonicalize_smiles)
    df = df.dropna(subset=["smiles"])
    df["inchikey"] = df["smiles"].map(_inchikey)
    df = df.dropna(subset=["inchikey"])

    n_before = len(df)
    df = df.groupby("inchikey", as_index=False).agg(smiles=("smiles", "first"), y=("y", "mean"))
    n_merged = n_before - len(df)
    df = df.reset_index(drop=True)

    smiles = df["smiles"].tolist()
    tr, va, te = split_dataset(smiles, split, fracs[0], fracs[1], fracs[2], seed, smiles_list=smiles)
    parts = {"train": list(tr.indices), "valid": list(va.indices), "test": list(te.indices)}
    for part, idx in parts.items():
        df.iloc[idx][["smiles", "y"]].to_csv(csv_dir / f"{part}.csv", index=False)

    print(
        f"[{name}] {len(df)} molecules ({n_merged} dup InChIKey merged) | "
        f"{split} split train/valid/test = {len(parts['train'])}/{len(parts['valid'])}/{len(parts['test'])} | "
        f"y mean={df['y'].mean():.3f} std={df['y'].std():.3f} range=[{df['y'].min():.3f}, {df['y'].max():.3f}]"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default="data/deepchem_molnet")
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--n-mad", type=float, default=5.0,
                    help="modified z-score threshold for flagging extreme targets")
    ap.add_argument("--drop-outliers", action="store_true",
                    help="REMOVE flagged extremes instead of only reporting them. This "
                         "changes the benchmark: RMSE falls because the hardest molecules "
                         "are gone, and the numbers stop being comparable to any published "
                         "MoleculeNet result. Off by default for that reason")
    ap.add_argument("--split", default="scaffold", choices=["scaffold", "random"])
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--force", action="store_true", help="re-download even if the raw CSV is cached")
    args = ap.parse_args()

    parent = Path(args.output_dir)
    for name in args.datasets:
        download_one(name, parent, args.split, (0.8, 0.1, 0.1), args.seed, args.force,
                     n_mad=args.n_mad, drop_outliers=args.drop_outliers)


if __name__ == "__main__":
    main()
