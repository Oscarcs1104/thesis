"""Download QM9's SMILES and report how much of the evaluation sets it reaches.

    python data_pipeline/download_qm9.py

Writes data/qm9.csv with one `smiles` column, then prints the heavy-atom distribution
next to MOSES and the three MoleculeNet sets, because the only reason to add this corpus
is coverage and the number should be visible before anything is trained on it.

Why QM9 and not more ZINC. Measured on 20k samples of each:

    MOSES       17 / 21 / 25 heavy atoms (p5/p50/p95), MW 300
    ZINC 250k   18 / 22 / 25                            MW 315
    FreeSolv     4 /  8 / 18                            MW 120
    ESOL         4 / 12 / 25                            MW 184
    Lipo        15 / 27 / 38                            MW 388

MOSES is ZINC Clean Leads filtered to MW 250-350, so ZINC at any scale sits in the same
narrow band -- ten million more of them would add no coverage. What that band misses is
both ends: FreeSolv lies entirely below it, half of ESOL lies below it, and more than
half of Lipophilicity lies above it. QM9 is at most 9 heavy atoms, which is FreeSolv's
regime exactly and ESOL's lower half. It does nothing for Lipophilicity; that end needs
a corpus of larger molecules, ChEMBL being the obvious one since Lipophilicity comes
from it.

QM9's limit to state plainly: it contains only C, N, O and F. FreeSolv has chlorine,
bromine and sulfur, so this covers the size regime and not all of the chemistry. Expect
part of the gap to close, not all of it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

QM9_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(ROOT / "data" / "qm9.csv"))
    p.add_argument("--url", default=QM9_URL)
    p.add_argument("--force", action="store_true", help="re-download even if the file is there")
    p.add_argument("--no-profile", action="store_true", help="skip the coverage comparison")
    return p.parse_args()


def heavy_atom_profile(smiles, sample: int = 20000):
    """(n, p5, p50, p95, median MW) over a sample, or None if nothing parses."""
    import numpy as np
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors

    RDLogger.DisableLog("rdApp.*")
    ha, mw = [], []
    for s in list(smiles)[:sample]:
        m = Chem.MolFromSmiles(str(s))
        if m is not None:
            ha.append(m.GetNumHeavyAtoms())
            mw.append(Descriptors.MolWt(m))
    if not ha:
        return None
    p5, p50, p95 = np.percentile(ha, [5, 50, 95])
    return len(ha), p5, p50, p95, float(np.median(mw))


def profile_everything(qm9_smiles) -> None:
    sources = [("QM9", qm9_smiles)]
    for label, rel, col in (
        ("MOSES", "data/moses/corpus.csv", "smiles"),
        ("FreeSolv", "data/deepchem_molnet/freesolv/csv/train.csv", "smiles"),
        ("ESOL", "data/deepchem_molnet/delaney/csv/train.csv", "smiles"),
        ("Lipo", "data/deepchem_molnet/lipo/csv/train.csv", "smiles"),
    ):
        path = ROOT / rel
        if path.exists():
            sources.append((label, pd.read_csv(path)[col].astype(str).tolist()))
        else:
            print(f"  ({label}: {rel} not there, skipped)")

    print(f"\n  {'':12}{'n':>8}{'p5':>7}{'p50':>7}{'p95':>7}{'MW mediana':>13}")
    profiles = {}
    for label, smiles in sources:
        prof = heavy_atom_profile(smiles)
        if prof is None:
            continue
        profiles[label] = prof
        n, p5, p50, p95, mw = prof
        print(f"  {label:<12}{n:>8,}{p5:>7.0f}{p50:>7.0f}{p95:>7.0f}{mw:>13.0f}")

    # The number the decision actually rests on: how much of each evaluation set falls
    # inside the corpus's own range. A percentile table invites eyeballing; this does not.
    if "MOSES" not in profiles:
        return
    import numpy as np
    from rdkit import Chem

    lo, hi = profiles["MOSES"][1], profiles["MOSES"][3]
    qm9_lo, qm9_hi = profiles["QM9"][1], profiles["QM9"][3]
    print(f"\n  fraccion de cada conjunto dentro del rango [p5, p95] del corpus:")
    print(f"  {'':12}{'MOSES':>18}{'QM9':>10}{'union':>10}")
    for label, smiles in sources[1:]:
        if label == "MOSES":
            continue
        ha = np.array([m.GetNumHeavyAtoms() for m in
                       (Chem.MolFromSmiles(str(s)) for s in smiles) if m is not None])
        if not len(ha):
            continue
        in_moses = ((ha >= lo) & (ha <= hi)).mean()
        in_qm9 = ((ha >= qm9_lo) & (ha <= qm9_hi)).mean()
        in_either = (((ha >= lo) & (ha <= hi)) | ((ha >= qm9_lo) & (ha <= qm9_hi))).mean()
        print(f"  {label:<12}{in_moses:>17.1%}{in_qm9:>10.1%}{in_either:>10.1%}")
    print("\n  'union' es lo que ganaria el corpus combinado. Lo que quede bajo en la")
    print("  tercera columna solo lo tapa un corpus de moleculas mas grandes.")


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and not args.force:
        df = pd.read_csv(out)
        print(f"{out} already there ({len(df):,} molecules). --force to re-download.")
    else:
        print(f"Downloading QM9 from {args.url} ...")
        raw = pd.read_csv(args.url)
        col = next((c for c in raw.columns if c.lower() in ("smiles", "canonical_smiles")), None)
        if col is None:
            raise SystemExit(f"no SMILES column in {args.url} (columns: {list(raw.columns)})")
        # SMILES only. The corpus builder canonicalizes, deduplicates and computes its own
        # RDKit labels, so QM9's quantum properties would go unused and are not carried.
        df = pd.DataFrame({"smiles": raw[col].astype(str)})
        before = len(df)
        df = df.drop_duplicates(subset="smiles").reset_index(drop=True)
        df.to_csv(out, index=False)
        print(f"Wrote {out}: {len(df):,} molecules ({before - len(df):,} duplicate strings dropped)")

    if not args.no_profile:
        profile_everything(df["smiles"].tolist())
        print("\nNext:  python data_pipeline/moses.py --extra-csv data/qm9.csv --force")


if __name__ == "__main__":
    main()
