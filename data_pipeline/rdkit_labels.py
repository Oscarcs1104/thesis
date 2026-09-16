"""Block 1b -- exact property labels for the MOSES corpus, computed with RDKit.

Four properties, all deterministic functions of the structure:

    logP   Crippen.MolLogP    octanol/water partition -- the headline conditioning target
    TPSA   rdMolDescriptors   topological polar surface area
    QED    QED.qed            quantitative estimate of drug-likeness, in [0, 1]
    MW     Descriptors.MolWt  molecular weight

Why this matters more than it looks: these are an *oracle*, not a model. Block 3 asks the
generator for a target logP and then measures the real logP of what came out with the same
function -- no learned predictor in the loop, so no circularity, no label noise, and no
out-of-distribution predictor. The previous evaluation scored the generator with the
generator's own regression head, which is what made its numbers unusable.

Writes, under --corpus-dir:
    labels.npy       float32 [N, 4], column order = PROPERTIES
    labels_meta.json per-property mean / std / percentiles, and the bin edges used later

Usage:
    python data_pipeline/rdkit_labels.py --corpus-dir data/moses
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROPERTIES = ["logp", "tpsa", "qed", "mw"]


def label_one(smiles: str) -> Optional[Tuple[float, float, float, float]]:
    """The four oracle values for one SMILES, or None if RDKit refuses it.

    Public because eval_oracle.py scores generated molecules with it. Sharing the
    function rather than reimplementing the four calls is the point: a change here
    cannot leave evaluation measuring a different quantity than the corpus labels.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return (
            float(Crippen.MolLogP(mol)),
            float(rdMolDescriptors.CalcTPSA(mol)),
            float(QED.qed(mol)),
            float(Descriptors.MolWt(mol)),
        )
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus-dir", type=str, default="data/moses")
    p.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    p.add_argument("--chunksize", type=int, default=2000)
    p.add_argument("--n-bins", type=int, default=20, help="quantile bins per property (for the prefix tokens)")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    corpus_dir = ROOT / args.corpus_dir if not Path(args.corpus_dir).is_absolute() else Path(args.corpus_dir)
    labels_path = corpus_dir / "labels.npy"
    if labels_path.exists() and not args.force:
        print(f"{labels_path} already exists. Use --force to recompute.")
        return

    corpus = pd.read_csv(corpus_dir / "corpus.csv")
    smiles: List[str] = corpus["smiles"].astype(str).tolist()
    print(f"Labelling {len(smiles)} molecules on {args.workers} workers...")

    start = time.time()
    results: List[Optional[Tuple[float, ...]]] = []
    if args.workers <= 1:
        for i, s in enumerate(smiles, 1):
            results.append(label_one(s))
            if i % 100_000 == 0:
                print(f"  {i}/{len(smiles)} ({time.time() - start:.0f}s)", flush=True)
    else:
        with mp.Pool(args.workers) as pool:
            for i, r in enumerate(pool.imap(label_one, smiles, chunksize=args.chunksize), 1):
                results.append(r)
                if i % 100_000 == 0:
                    print(f"  {i}/{len(smiles)} ({time.time() - start:.0f}s)", flush=True)

    # A failure here would desynchronize labels from the token array, so NaN-fill and
    # report instead of dropping rows: every downstream index must keep pointing at the
    # same molecule it does in corpus.csv / selfies_tokens.npy.
    n_failed = sum(1 for r in results if r is None)
    labels = np.array(
        [r if r is not None else (np.nan,) * len(PROPERTIES) for r in results],
        dtype=np.float32,
    )
    if n_failed:
        print(f"  [warn] {n_failed} molecules failed to label; their rows are NaN "
              f"(mine_pairs.py drops them)")

    np.save(labels_path, labels)

    meta = {"properties": PROPERTIES, "n": int(labels.shape[0]), "n_failed": n_failed, "per_property": {}}
    print(f"\n{'property':<8} {'mean':>9} {'std':>9} {'p1':>9} {'p50':>9} {'p99':>9}")
    print("-" * 58)
    for i, name in enumerate(PROPERTIES):
        col = labels[:, i]
        col = col[np.isfinite(col)]
        # Quantile bins, computed once here so the trainer, the sampler and the evaluator
        # all agree on what "bin 7" means. Duplicate edges are collapsed (QED piles up).
        edges = np.unique(np.quantile(col, np.linspace(0, 1, args.n_bins + 1)))
        meta["per_property"][name] = {
            "mean": float(col.mean()), "std": float(col.std()),
            "min": float(col.min()), "max": float(col.max()),
            "p1": float(np.percentile(col, 1)), "p50": float(np.percentile(col, 50)),
            "p99": float(np.percentile(col, 99)),
            "bin_edges": [float(e) for e in edges],
            "n_bins": int(len(edges) - 1),
        }
        print(f"{name:<8} {col.mean():>9.3f} {col.std():>9.3f} {np.percentile(col, 1):>9.3f} "
              f"{np.percentile(col, 50):>9.3f} {np.percentile(col, 99):>9.3f}")

    (corpus_dir / "labels_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nWrote {labels_path} {labels.shape} and labels_meta.json")
    print("Next: python data_pipeline/mine_pairs.py --corpus-dir", args.corpus_dir)


if __name__ == "__main__":
    main()
