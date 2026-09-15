"""Summarize a built MOSES corpus: sizes, leakage, delta spread, scaffold structure.

Reads only the cached artifacts, so it is instant and safe to re-run.

The scaffold section answers the design question the pipeline hangs on: how much of a
molecule its Murcko scaffold already accounts for. If the scaffold covers almost every
heavy atom there is little left to generate and the task collapses toward copying; the
generic framework (atom types erased) is then the harder, more generative alternative.

    python data_pipeline/report_corpus.py --corpus-dir data/moses
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _coverage(args):
    """(heavy atoms in scaffold, heavy atoms in molecule, generic-framework atoms)."""
    smiles, scaffold = args
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    n_mol = mol.GetNumHeavyAtoms()
    n_scaf = 0
    if scaffold:
        s = Chem.MolFromSmiles(scaffold)
        n_scaf = s.GetNumHeavyAtoms() if s is not None else 0
    n_generic = 0
    try:
        generic = MurckoScaffold.MakeScaffoldGeneric(mol)
        n_generic = generic.GetNumHeavyAtoms()
    except Exception:
        pass
    return n_scaf, n_mol, n_generic


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-dir", default="data/moses")
    ap.add_argument("--sample", type=int, default=20000, help="molecules sampled for the scaffold stats")
    ap.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    args = ap.parse_args()

    d = ROOT / args.corpus_dir if not Path(args.corpus_dir).is_absolute() else Path(args.corpus_dir)
    load = lambda name: json.loads((d / name).read_text(encoding="utf-8")) if (d / name).exists() else None
    meta, dedup, pairs_meta = load("meta.json"), load("dedup_report.json"), load("pairs_meta.json")

    print("=" * 72)
    print("CORPUS")
    print("=" * 72)
    if meta:
        print(f"  molecules      {meta['n_molecules']:>10,}   (raw {meta['n_raw']:,}, "
              f"rejected {meta['n_rejected']:,}, dup {meta['n_internal_duplicates']:,})")
        print(f"  SELFIES vocab  {meta['vocab_size']:>10}   max_len {meta['max_len']} "
              f"(p50 {meta['length_percentiles']['50']:.0f}, {meta['n_truncated']:,} truncated)")
        print(f"  deduplicated   {str(meta.get('deduplicated_against_eval', '?')):>10}")

    if dedup:
        print("\n" + "=" * 72)
        print("LEAKAGE vs the esol/freesolv/lipo TEST splits")
        print("=" * 72)
        r = dedup["removed_exact_inchikey"]
        print(f"  removed (exact InChIKey)   {r['total']:>8,}   {r['per_dataset']}")
        print(f"  test molecules checked     {dedup['eval_test_molecules_total_unique']:>8,}   "
              f"{dedup['eval_test_molecules']}")
        so = dedup["scaffold_overlap"]
        print(f"  shared scaffolds           {so['n_shared_scaffolds']:>8,}   covering "
              f"{so['n_corpus_molecules_with_shared_scaffold']:,} molecules "
              f"({so['fraction_of_corpus']:.2%}) -- {so['policy']}")

    if pairs_meta:
        print("\n" + "=" * 72)
        print("ANALOG PAIRS")
        print("=" * 72)
        print(f"  pairs {pairs_meta['n_pairs']:,}  (identity {pairs_meta['n_identity']:,})")
        print(f"\n  {'delta':<8} {'std':>9} {'p1':>9} {'p99':>9} {'>1sd':>7}  eval range (p5..p95)")
        print("  " + "-" * 68)
        for name, s in pairs_meta["delta_stats"].items():
            rng = s.get("eval_deltas_p5_p95")
            rng_s = "[" + ", ".join(f"{p:+.2f}" for p in rng) + "]" if rng else "(re-run mine_pairs)"
            big = s.get("frac_abs_gt_1_std", s.get("frac_abs_gt_1", float("nan")))
            print(f"  {name:<8} {s['std']:>9.3f} {s['p1']:>9.3f} {s['p99']:>9.3f} {big:>7.1%}  {rng_s}")

    # ------------------------------------------------------------------ scaffolds
    corpus = pd.read_csv(d / "corpus.csv")
    if "scaffold" not in corpus.columns:
        print("\n(no scaffold column -- rebuild the corpus with data_pipeline/moses.py)")
        return

    scaffolds = corpus["scaffold"].fillna("").astype(str)
    counts = Counter(scaffolds)
    sizes = np.array(sorted(counts.values(), reverse=True))
    n_empty = counts.get("", 0)

    print("\n" + "=" * 72)
    print("SCAFFOLDS  (decides scaffold-conditioned generation vs. analog pairs)")
    print("=" * 72)
    print(f"  distinct scaffolds         {len(counts):>10,}  for {len(scaffolds):,} molecules")
    print(f"  molecules per scaffold     mean {sizes.mean():>5.1f}   median {np.median(sizes):>3.0f}   "
          f"max {sizes.max():,}")
    print(f"  singletons                 {int((sizes == 1).sum()):>10,}  "
          f"({(sizes == 1).sum() / len(sizes):.1%} of scaffolds, "
          f"{(sizes == 1).sum() / len(scaffolds):.1%} of molecules)")
    print(f"  acyclic (empty scaffold)   {n_empty:>10,}")
    for k in (2, 5, 10, 50):
        n_mols = int(sizes[sizes >= k].sum())
        print(f"  in a bucket of >= {k:<3d}        {n_mols:>10,}  ({n_mols / len(scaffolds):.1%} of molecules)")

    sample = corpus.sample(min(args.sample, len(corpus)), random_state=0)
    items = list(zip(sample["smiles"].astype(str), sample["scaffold"].fillna("").astype(str)))
    print(f"\n  computing atom coverage on {len(items):,} sampled molecules...")
    with mp.Pool(args.workers) as pool:
        res = [r for r in pool.map(_coverage, items, chunksize=200) if r is not None]
    scaf = np.array([r[0] for r in res], dtype=float)
    mol = np.array([r[1] for r in res], dtype=float)
    frac = np.divide(scaf, np.maximum(mol, 1))

    print(f"  heavy atoms: molecule {mol.mean():.1f} avg, scaffold {scaf.mean():.1f} avg")
    print(f"  scaffold covers {frac.mean():.1%} of the molecule "
          f"(p25 {np.percentile(frac, 25):.0%}, p75 {np.percentile(frac, 75):.0%})")
    print(f"  atoms left to generate: {(mol - scaf).mean():.1f} on average")
    print()
    if frac.mean() > 0.8:
        print("  >80% covered: conditioning on the Murcko scaffold leaves little to invent, so")
        print("  the task drifts toward copying. Prefer the GENERIC FRAMEWORK (atom types")
        print("  erased, topology kept), which puts the property back in charge.")
    elif frac.mean() < 0.5:
        print("  <50% covered: the Murcko scaffold underspecifies the molecule heavily. Good")
        print("  entropy for scaffold-conditioned generation; the generic framework would")
        print("  likely be too loose a constraint.")
    else:
        print("  50-80% covered: the Murcko scaffold is a reasonable constraint. Start there,")
        print("  and switch to the generic framework if generated novelty comes out low.")


if __name__ == "__main__":
    main()
