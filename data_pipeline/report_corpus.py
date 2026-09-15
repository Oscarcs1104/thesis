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


def _framework_only(scaffold: str) -> str:
    """Murcko scaffold SMILES -> generic framework SMILES (atom types and bond orders
    erased, topology kept). Empty string for acyclic molecules, which have no scaffold."""
    if not scaffold:
        return ""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    from rdkit.Chem.Scaffolds import MurckoScaffold

    m = Chem.MolFromSmiles(scaffold)
    if m is None:
        return ""
    try:
        return Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(m))
    except Exception:
        return ""


def _coverage(args):
    """(heavy atoms in scaffold, heavy atoms in molecule, generic framework SMILES).

    The generic framework is MakeScaffoldGeneric applied to the SCAFFOLD, not to the
    molecule: it keeps the skeleton's topology and erases atom types and bond orders.
    So it has the same atom count as the Murcko scaffold but carries far less
    information -- the decoder must decide which positions are N, O or S and where the
    double bonds go, and heteroatoms are exactly what drives logP and TPSA.
    """
    smiles, scaffold = args
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    n_mol = mol.GetNumHeavyAtoms()
    n_scaf, generic_smiles = 0, ""
    if scaffold:
        s = Chem.MolFromSmiles(scaffold)
        if s is not None:
            n_scaf = s.GetNumHeavyAtoms()
            try:
                generic_smiles = Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(s))
            except Exception:
                generic_smiles = ""
    return n_scaf, n_mol, generic_smiles


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-dir", default="data/moses")
    ap.add_argument("--sample", type=int, default=20000, help="molecules sampled for the scaffold stats")
    ap.add_argument("--frameworks", action="store_true",
                    help="compute the generic framework for EVERY molecule and cache it. "
                         "Required for real bucket statistics -- a sample cannot give them")
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

    # Bucket occupancy MUST be measured on the whole corpus. Counting it inside a random
    # 20k sample of 1.9M is meaningless: a scaffold with 10 corpus molecules shows up
    # twice in such a sample with probability ~0.1%, so almost everything looks unique.
    # Only the ratio of distinct counts survives sampling, so that is all the sample is
    # used for here; the occupancy numbers come from --frameworks over every molecule.
    gen_sample = Counter(r[2] for r in res if r[2])
    scaf_sample = Counter(s for s, _ in items if s)
    ratio = len(scaf_sample) / max(len(gen_sample), 1)
    print(f"\n  erasing atom types and bond orders collapses {ratio:.1f} Murcko scaffolds into "
          f"one generic framework")

    fw_path = d / "frameworks.csv"
    if args.frameworks:
        print(f"  computing generic frameworks for all {len(corpus):,} molecules...")
        with mp.Pool(args.workers) as pool:
            fws = pool.map(_framework_only, corpus["scaffold"].fillna("").astype(str).tolist(),
                           chunksize=2000)
        pd.DataFrame({"framework": fws}).to_csv(fw_path, index=False)
        print(f"  cached to {fw_path}")

    if fw_path.exists():
        fws = pd.read_csv(fw_path)["framework"].fillna("").astype(str)
        fw_counts = Counter(f for f in fws if f)
        fw_sizes = np.array(sorted(fw_counts.values(), reverse=True))
        print("\n  constraint strength over the FULL corpus "
              "(molecules sharing one constraint = entropy the property must resolve):")
        print(f"    {'constraint':<22} {'distinct':>10} {'mean':>8} {'median':>8} {'>=10':>16}")
        for label, c_sizes, total in (("Murcko scaffold", sizes, len(scaffolds)),
                                      ("generic framework", fw_sizes, len(fws))):
            big = int(c_sizes[c_sizes >= 10].sum())
            print(f"    {label:<22} {len(c_sizes):>10,} {c_sizes.mean():>8.1f} "
                  f"{np.median(c_sizes):>8.0f} {big / total:>15.1%}")
    else:
        print(f"\n  (pass --frameworks to measure framework buckets over the whole corpus;"
              f" the sample above cannot)")
    print()
    if frac.mean() > 0.7:
        print(f"  The Murcko scaffold already fixes {frac.mean():.0%} of the molecule and leaves only")
        print(f"  {(mol - scaf).mean():.1f} atoms to invent. That is decoration, not generation.")
        print("  The GENERIC FRAMEWORK keeps the same skeleton size but erases atom types and")
        print("  bond orders, so the decoder must place the heteroatoms -- which is exactly what")
        print("  determines logP and TPSA. Same constraint shape, far more entropy.")
    else:
        print("  The Murcko scaffold leaves substantial freedom; it is a usable constraint on")
        print("  its own, and the generic framework would likely be too loose.")


if __name__ == "__main__":
    main()
