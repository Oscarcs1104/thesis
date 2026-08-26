"""Download ZINC15 (1M) via DeepChem and export two separate SMILES-only CSVs.

DeepChem's load_zinc15 only supports fixed dataset_size buckets: '250K', '1M',
'10M', '270M' -- there is no '500K' bucket to download directly. This script
downloads the '1M' bucket once and writes it out as two independent files:
  - data/zinc15_1M.csv    (all downloaded molecules)
  - data/zinc15_500K.csv  (a random subsample, same format as zinc15_250K.csv)

Usage:
  python data_pipeline/download_zinc15.py
  python data_pipeline/download_zinc15.py --subsample-size 500000 --seed 2025
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ZINC15 1M via DeepChem and export a 1M + a subsampled CSV")
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent.parent / "data"), help="Where to write the CSVs")
    parser.add_argument("--cache-dir", type=str, default=None, help="DeepChem raw/featurized cache dir (default: <output-dir>/zinc15_cache)")
    parser.add_argument("--dataset-size", type=str, default="1M", choices=["250K", "1M", "10M", "270M"], help="DeepChem ZINC15 bucket to download")
    parser.add_argument("--subsample-size", type=int, default=500_000, help="Size of the extra random subsample written as a separate CSV")
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


def _write_smiles_csv(smiles_list: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["smiles"])
        for smi in smiles_list:
            writer.writerow([smi])


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else output_dir / "zinc15_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # DeepChem's CSVLoader uses id_field="zinc_id" and feature_field="smiles" internally, so
    # dataset.ids/dataset.X (via the Dataset API) don't reliably give back SMILES strings --
    # instead, trigger the download once, then read the raw CSV it caches directly.
    raw_csv_path = cache_dir / f"zinc15_{args.dataset_size}_2D.csv"
    if not raw_csv_path.exists():
        try:
            import deepchem as dc  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "deepchem is not installed in the active environment. Install it first or run inside the project's environment."
            ) from exc

        print(f"Downloading ZINC15 {args.dataset_size} (2D) via DeepChem -- this can take a while the first time...")
        dc.molnet.load_zinc15(
            featurizer=dc.feat.RawFeaturizer(smiles=True),
            splitter=None,
            dataset_size=args.dataset_size,
            dataset_dimension="2D",
            reload=True,
            data_dir=str(cache_dir),
            save_dir=str(cache_dir),
        )

    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Expected raw ZINC15 CSV at {raw_csv_path} after download, but it's missing.")

    print(f"Reading SMILES column directly from {raw_csv_path}")
    smiles_list: list[str] = []
    with raw_csv_path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            smi = row.get("smiles")
            if smi:
                smiles_list.append(smi)
    print(f"Loaded {len(smiles_list)} molecules")

    full_path = output_dir / f"zinc15_{args.dataset_size}.csv"
    _write_smiles_csv(smiles_list, full_path)
    print(f"Wrote {len(smiles_list)} molecules to {full_path}")

    subsample_size = min(args.subsample_size, len(smiles_list))
    random.seed(args.seed)
    subsample = random.sample(smiles_list, subsample_size)
    subsample_path = output_dir / f"zinc15_{subsample_size // 1000}K.csv"
    _write_smiles_csv(subsample, subsample_path)
    print(f"Wrote {len(subsample)} molecules to {subsample_path}")


if __name__ == "__main__":
    main()
