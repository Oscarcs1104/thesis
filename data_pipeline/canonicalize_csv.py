"""Canonicalize the SMILES column of a CSV into a new file, and report duplicates
that canonicalization reveals (different raw SMILES for the same molecule).

Never modifies the input file.

Usage:
  python data_pipeline/canonicalize_csv.py --csv data/esol.csv --out data/esol_canonical.csv
"""
from __future__ import annotations

import argparse
import csv
import gzip
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data_pipeline.convert_smiles_to_pyg import canonicalize_smiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonicalize a CSV's SMILES column and report duplicates")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--smiles-col", type=str, default="smiles")
    parser.add_argument("--out", type=str, required=True, help="Path to the new canonicalized CSV")
    parser.add_argument("--dedupe", action="store_true", help="Also write a duplicate-free copy next to --out")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    open_f = gzip.open if csv_path.suffix.lower() == ".gz" else open

    with open_f(csv_path, "rt", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames or args.smiles_col not in fieldnames:
            raise ValueError(f"Column '{args.smiles_col}' not found. Available columns: {fieldnames}")
        rows = list(reader)

    out_rows = []
    invalid = 0
    seen: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        canonical = canonicalize_smiles(row[args.smiles_col])
        if canonical is None:
            invalid += 1
            continue
        new_row = dict(row)
        new_row[args.smiles_col] = canonical
        seen[canonical].append(len(out_rows))
        out_rows.append(new_row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    duplicates = {smi: idxs for smi, idxs in seen.items() if len(idxs) > 1}
    dup_rows = sum(len(idxs) for idxs in duplicates.values())

    print(f"Wrote {len(out_rows)} canonicalized rows to {out_path}")
    if invalid:
        print(f"Skipped {invalid} rows with unparsable SMILES")
    print(f"Duplicate canonical SMILES: {len(duplicates)} molecules, {dup_rows} rows involved")

    if args.dedupe:
        added: set[str] = set()
        dedup_rows = []
        for row in out_rows:
            canonical = row[args.smiles_col]
            if canonical in added:
                continue
            added.add(canonical)
            dedup_rows.append(row)
        dedup_path = out_path.with_name(out_path.stem + "_dedup" + out_path.suffix)
        with dedup_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(dedup_rows)
        print(f"Wrote {len(dedup_rows)} deduplicated rows to {dedup_path}")


if __name__ == "__main__":
    main()
