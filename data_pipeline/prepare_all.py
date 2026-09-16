"""One command to (re)build every dataset artifact deterministically.

  1. download ESOL / FreeSolv / Lipophilicity (raw MoleculeNet CSVs) and write a
     frozen split to data/deepchem_molnet/<name>/csv/{train,valid,test}.csv
     (--split random by default; --split scaffold for the harder partition)
  2. warm the PyG graph cache for every split CSV (OGB-style features)
  3. warm the graph cache for data/zinc15_250K.csv -- OPTIONAL, --skip-zinc turns it
     off. It fed the superseded pseudo-labelling generator; the conditional
     generator pretrains on MOSES instead and nothing current reads it.

No deepchem / tensorflow needed. Nothing here is committed -- run it on a fresh
clone before training.

    python data_pipeline/prepare_all.py
    python data_pipeline/prepare_all.py --skip-zinc        # what the current pipeline needs
    python data_pipeline/prepare_all.py --skip-download    # just rebuild caches
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data_pipeline.data import load_graph_dataset

_MOLNET = {"delaney": "esol", "freesolv": "freesolv", "lipo": "lipo"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default="data/deepchem_molnet")
    ap.add_argument("--zinc-csv", default="data/zinc15_250K.csv")
    ap.add_argument("--split", default="random", choices=["random", "scaffold"],
                    help="MoleculeNet partition. Random is what the MoleculeNet paper "
                         "recommends for these three physical-chemistry regression sets; "
                         "scaffold is harder and is the convention for the biological "
                         "classification sets. The corpus deduplication covers every split "
                         "of all three datasets, so changing this does not affect it")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--merge-duplicates", action="store_true",
                    help="merge molecules sharing an InChIKey and average their targets. "
                         "Off by default: it makes a cleaner dataset and a different one, "
                         "and these numbers are meant to be comparable with published "
                         "MoleculeNet results measured on the raw files")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--n-mad", type=float, default=5.0,
                    help="modified z-score threshold for flagging extreme targets")
    ap.add_argument("--drop-outliers", action="store_true",
                    help="REMOVE flagged extremes rather than only reporting them. Doing so "
                         "changes the benchmark and the numbers stop being comparable to "
                         "published MoleculeNet results; off by default")
    ap.add_argument("--skip-zinc", action="store_true",
                    help="skip the ZINC15 graph cache. It fed the superseded pseudo-labelling "
                         "generator; the conditional generator pretrains on MOSES "
                         "(data_pipeline/moses.py) instead, so nothing current reads it")
    ap.add_argument("--force", action="store_true", help="re-download raw CSVs even if cached")
    args = ap.parse_args()

    out_base = Path(args.output_dir)

    if not args.skip_download:
        from data_pipeline.download_molnet import download_one

        for molnet_name in _MOLNET:
            download_one(molnet_name, out_base, split=args.split, seed=args.seed,
                         force=args.force, n_mad=args.n_mad,
                         drop_outliers=args.drop_outliers,
                         merge_duplicates=args.merge_duplicates)

    print("\n== warming graph caches ==")
    for molnet_name in _MOLNET:
        for split in ("train", "valid", "test"):
            csv_path = out_base / molnet_name / "csv" / f"{split}.csv"
            if csv_path.exists():
                graphs = load_graph_dataset(str(csv_path))
                print(f"  {csv_path}: {len(graphs)} graphs")
            else:
                print(f"  MISSING {csv_path} (run without --skip-download)")

    if args.skip_zinc:
        print("  skipping ZINC15 (--skip-zinc): superseded by the MOSES corpus")
    elif Path(args.zinc_csv).exists():
        graphs = load_graph_dataset(args.zinc_csv)
        print(f"  {args.zinc_csv}: {len(graphs)} graphs")
    else:
        print(f"  MISSING {args.zinc_csv} -- run data_pipeline/download_zinc15.py")

    print("\nDone. Next:")
    print("  predictor half : python thesis_model/benchmark/run_baselines.py")
    print("  generator half : sbatch scripts/slurm/block1_data.sbatch   (builds the MOSES corpus)")


if __name__ == "__main__":
    main()
