"""One command to (re)build every dataset artifact deterministically.

  1. download ESOL / FreeSolv / Lipophilicity (raw MoleculeNet CSVs) and write a
     frozen SCAFFOLD split to data/deepchem_molnet/<name>/csv/{train,valid,test}.csv
  2. warm the PyG graph cache for every split CSV (OGB-style features)
  3. warm the graph cache for data/zinc15_250K.csv (generator pretraining pool)

No deepchem / tensorflow needed. Nothing here is committed -- run it on a fresh
clone before training.

    python data_pipeline/prepare_all.py
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
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-download raw CSVs even if cached")
    args = ap.parse_args()

    out_base = Path(args.output_dir)

    if not args.skip_download:
        from data_pipeline.download_molnet import download_one

        for molnet_name in _MOLNET:
            download_one(molnet_name, out_base, split="scaffold", seed=args.seed, force=args.force)

    print("\n== warming graph caches ==")
    for molnet_name in _MOLNET:
        for split in ("train", "valid", "test"):
            csv_path = out_base / molnet_name / "csv" / f"{split}.csv"
            if csv_path.exists():
                graphs = load_graph_dataset(str(csv_path))
                print(f"  {csv_path}: {len(graphs)} graphs")
            else:
                print(f"  MISSING {csv_path} (run without --skip-download)")

    if Path(args.zinc_csv).exists():
        graphs = load_graph_dataset(args.zinc_csv)
        print(f"  {args.zinc_csv}: {len(graphs)} graphs")
    else:
        print(f"  MISSING {args.zinc_csv} -- run data_pipeline/download_zinc15.py")

    print("\nDone. Train with e.g.:")
    print("  python training/train.py --dataset-dir data/deepchem_molnet/delaney --graph-backbone gin --seeds 2025 2026 2027")
    print("  python scripts/run_baselines.py")


if __name__ == "__main__":
    main()
