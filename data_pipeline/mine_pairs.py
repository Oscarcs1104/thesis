"""Block 1c -- mine (source, target, delta) triples: the training signal for the generator.

This is what breaks the autoencoder collapse. Training on (M -> M, y=f(M)) puts the target
molecule inside the decoder's own memory, so reproducing it never requires reading the
conditioning token and the gradient has no reason to teach the decoder to use it. Here the
encoder sees M_a, the condition is a *delta*, and the decoder must emit a different
molecule M_b:

    encoder input : M_a           e.g. chlorobenzene   logP 2.84
    condition     : delta logP    -0.57
    decoder target: M_b           e.g. fluorobenzene   logP 2.27

M_b is not in the memory, so the delta is the only thing that says "swap Cl for F, not Br".
H(target | memory, delta) > 0 by construction, and the encoder stays load-bearing: without
it the decoder doesn't know what to modify. It also fixes the train/inference mismatch --
delta = -1 at sampling time is something the model saw thousands of times, unlike the old
"y = true_value +/- 1 sigma" combination, which never occurred in training.

How pairs are found: bucket by Bemis-Murcko scaffold (molecules sharing a scaffold are the
analogs we want, and it turns an impossible 1.9M x 1.9M comparison into something linear),
then keep pairs whose ECFP4 Tanimoto falls in a band. Below --tanimoto-min it is a
different molecule rather than a modification, and the encoder can't help; above
--tanimoto-max it is a near-copy and teaches nothing.

Identity pairs (M -> M, delta = 0) are mixed in at --identity-frac. They pin down the
semantics of delta = 0 ("change nothing") and give a free sanity check at evaluation time:
ask for 0 and the seed should come back.

Writes, under --corpus-dir:
    pairs.npy        int32   [M, 2]  source index, target index into corpus.csv
    pair_deltas.npy  float32 [M, 4]  labels[target] - labels[source], column order = PROPERTIES
    pairs_meta.json  counts, settings, and the delta histogram

Usage:
    python data_pipeline/mine_pairs.py --corpus-dir data/moses
    python data_pipeline/mine_pairs.py --corpus-dir data/moses --balance-delta logp
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.rdkit_labels import PROPERTIES  # noqa: E402

_WORKER_CFG: dict = {}


def _init_worker(cfg: dict) -> None:
    global _WORKER_CFG
    _WORKER_CFG = cfg
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")


def _mine_chunk(chunk: Tuple[List[int], List[str]]) -> List[Tuple[int, int]]:
    """Find analog pairs inside one scaffold chunk. Returns ordered (source, target) pairs.

    Both directions of every found pair are emitted: the delta is signed, so (a, b) teaches
    the model to increase a property and (b, a) to decrease it. Training on only one
    direction would bias every conditioning token toward one sign.
    """
    indices, smiles = chunk
    n = len(indices)
    if n < 2:
        return []

    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.DataStructs import BulkTanimotoSimilarity

    cfg = _WORKER_CFG
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    fps, keep_pos = [], []
    for pos, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fps.append(gen.GetFingerprint(mol))
        keep_pos.append(pos)
    if len(fps) < 2:
        return []

    rng = np.random.default_rng(cfg["seed"] + indices[0])
    tmin, tmax = cfg["tanimoto_min"], cfg["tanimoto_max"]
    max_per_mol, n_candidates = cfg["max_pairs_per_mol"], cfg["candidates_per_mol"]

    found = set()
    m = len(fps)
    for i in range(m):
        # Sampling candidates rather than comparing against the whole bucket keeps the
        # cost linear: a popular scaffold can hold tens of thousands of molecules.
        if m - 1 <= n_candidates:
            cand = [j for j in range(m) if j != i]
        else:
            cand = rng.choice(m, size=n_candidates, replace=False)
            cand = [int(j) for j in cand if j != i]
        if not cand:
            continue
        sims = BulkTanimotoSimilarity(fps[i], [fps[j] for j in cand])
        hits = [(s, j) for s, j in zip(sims, cand) if tmin <= s <= tmax]
        if not hits:
            continue
        # Prefer the least similar admissible analogs: they carry the largest structural
        # (and therefore property) change, which is what the delta has to explain.
        hits.sort(key=lambda t: t[0])
        for _, j in hits[:max_per_mol]:
            a, b = indices[keep_pos[i]], indices[keep_pos[j]]
            found.add((a, b))
            found.add((b, a))
    return sorted(found)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus-dir", type=str, default="data/moses")
    p.add_argument("--tanimoto-min", type=float, default=0.5)
    p.add_argument("--tanimoto-max", type=float, default=0.95)
    p.add_argument("--max-pairs-per-mol", type=int, default=10)
    p.add_argument("--candidates-per-mol", type=int, default=200,
                   help="random candidates compared per molecule inside its scaffold bucket")
    p.add_argument("--max-bucket", type=int, default=5000,
                   help="large scaffold buckets are split into chunks of this size, for load balance")
    p.add_argument("--identity-frac", type=float, default=0.05,
                   help="fraction of total pairs that are (M -> M, delta 0)")
    p.add_argument("--balance-delta", type=str, default=None, choices=[None, *PROPERTIES],
                   help="flatten the delta histogram of this property by capping per-bin counts")
    p.add_argument("--balance-bins", type=int, default=20)
    p.add_argument("--balance-cap-quantile", type=float, default=0.9,
                   help="per-bin cap = this quantile of the bin occupancies. The effect is "
                        "very non-linear in how peaked the deltas are, so run once WITHOUT "
                        "--balance-delta, read the printed histogram, then tune this")
    p.add_argument("--max-pairs", type=int, default=None, help="hard cap on the final pair count")
    p.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def _build_chunks(scaffolds: Sequence[str], valid: np.ndarray, smiles: Sequence[str],
                  max_bucket: int, rng: np.random.Generator) -> List[Tuple[List[int], List[str]]]:
    buckets: Dict[str, List[int]] = defaultdict(list)
    for idx in np.flatnonzero(valid):
        buckets[scaffolds[idx]].append(int(idx))

    chunks = []
    for members in buckets.values():
        if len(members) < 2:
            continue
        if len(members) > max_bucket:
            members = list(members)
            rng.shuffle(members)
            for start in range(0, len(members), max_bucket):
                part = members[start: start + max_bucket]
                if len(part) >= 2:
                    chunks.append((part, [smiles[i] for i in part]))
        else:
            chunks.append((members, [smiles[i] for i in members]))
    # Biggest first: long jobs start early so the tail doesn't stall on one huge chunk.
    chunks.sort(key=lambda c: -len(c[0]))
    return chunks


def main() -> None:
    args = parse_args()
    corpus_dir = ROOT / args.corpus_dir if not Path(args.corpus_dir).is_absolute() else Path(args.corpus_dir)
    pairs_path = corpus_dir / "pairs.npy"
    if pairs_path.exists() and not args.force:
        print(f"{pairs_path} already exists. Use --force to rebuild.")
        return

    corpus = pd.read_csv(corpus_dir / "corpus.csv")
    if "scaffold" not in corpus.columns:
        raise SystemExit("corpus.csv has no 'scaffold' column -- rebuild it with data_pipeline/moses.py")
    labels = np.load(corpus_dir / "labels.npy")
    smiles = corpus["smiles"].astype(str).tolist()
    scaffolds = corpus["scaffold"].fillna("").astype(str).tolist()

    # Molecules whose labels failed are unusable: their delta would be NaN.
    valid = np.isfinite(labels).all(axis=1)
    print(f"Corpus: {len(smiles)} molecules, {int(valid.sum())} with complete labels")

    rng = np.random.default_rng(args.seed)
    chunks = _build_chunks(scaffolds, valid, smiles, args.max_bucket, rng)
    n_in_chunks = sum(len(c[0]) for c in chunks)
    print(f"Scaffold buckets -> {len(chunks)} chunks covering {n_in_chunks} molecules "
          f"(largest {len(chunks[0][0]) if chunks else 0}); "
          f"{int(valid.sum()) - n_in_chunks} are alone in their scaffold and yield no pair")

    cfg = {
        "seed": args.seed, "tanimoto_min": args.tanimoto_min, "tanimoto_max": args.tanimoto_max,
        "max_pairs_per_mol": args.max_pairs_per_mol, "candidates_per_mol": args.candidates_per_mol,
    }
    print(f"Mining pairs on {args.workers} workers "
          f"(Tanimoto in [{args.tanimoto_min}, {args.tanimoto_max}])...")
    start = time.time()
    collected: List[Tuple[int, int]] = []
    if args.workers <= 1:
        _init_worker(cfg)
        for i, chunk in enumerate(chunks, 1):
            collected.extend(_mine_chunk(chunk))
            if i % 500 == 0 or i == len(chunks):
                print(f"  {i}/{len(chunks)} chunks, {len(collected)} pairs ({time.time() - start:.0f}s)", flush=True)
    else:
        with mp.Pool(args.workers, initializer=_init_worker, initargs=(cfg,)) as pool:
            for i, res in enumerate(pool.imap_unordered(_mine_chunk, chunks, chunksize=8), 1):
                collected.extend(res)
                if i % 500 == 0 or i == len(chunks):
                    print(f"  {i}/{len(chunks)} chunks, {len(collected)} pairs ({time.time() - start:.0f}s)", flush=True)

    if not collected:
        if len(chunks) < 50:
            raise SystemExit(
                f"No pairs found, but only {len(chunks)} scaffold buckets had 2+ members out of "
                f"{int(valid.sum())} molecules. On a small --limit run that is arithmetic, not a "
                f"bug: MOSES holds hundreds of thousands of distinct Murcko scaffolds, so in a "
                f"small sample nearly every molecule is alone in its scaffold. Rebuild the corpus "
                f"with --limit 50000 or more."
            )
        raise SystemExit("No pairs found -- loosen --tanimoto-min or raise --candidates-per-mol.")
    pairs = np.asarray(collected, dtype=np.int32)
    deltas = (labels[pairs[:, 1]] - labels[pairs[:, 0]]).astype(np.float32)
    print(f"Mined {len(pairs)} analog pairs in {time.time() - start:.0f}s")

    # ---------------- optional delta balancing ----------------
    if args.balance_delta:
        col = PROPERTIES.index(args.balance_delta)
        d = deltas[:, col]
        lo, hi = np.percentile(d, 0.5), np.percentile(d, 99.5)
        edges = np.linspace(lo, hi, args.balance_bins + 1)
        bin_idx = np.clip(np.digitize(d, edges) - 1, 0, args.balance_bins - 1)
        counts = np.bincount(bin_idx, minlength=args.balance_bins)
        # The near-zero bins hold most of the mass; leaving them there lets the model score
        # well by always predicting "barely change it". Cap every bin at a quantile of the
        # bin occupancies: the median flattens hardest but can discard >90% of the pairs on
        # a peaked distribution, so the default is milder and the reduction is printed.
        cap = int(np.quantile(counts[counts > 0], args.balance_cap_quantile))
        keep = np.concatenate([
            rng.permutation(np.flatnonzero(bin_idx == b))[:cap] for b in range(args.balance_bins)
        ])
        keep.sort()
        before_ratio = counts.max() / max(counts[counts > 0].min(), 1)
        after_counts = np.bincount(bin_idx[keep], minlength=args.balance_bins)
        after_ratio = after_counts.max() / max(after_counts[after_counts > 0].min(), 1)
        print(f"Balancing on delta {args.balance_delta}: {len(pairs)} -> {len(keep)} pairs "
              f"({len(keep) / len(pairs):.0%} kept, cap {cap}/bin over {args.balance_bins} bins)")
        print(f"  bin imbalance {before_ratio:.0f}x -> {after_ratio:.1f}x")
        if len(keep) < 0.2 * len(pairs):
            print(f"  [warn] that discarded {1 - len(keep) / len(pairs):.0%} of the pairs; "
                  f"raise --balance-cap-quantile if the corpus ends up too small")
        pairs, deltas = pairs[keep], deltas[keep]

    if args.max_pairs and len(pairs) > args.max_pairs:
        keep = rng.permutation(len(pairs))[: args.max_pairs]
        keep.sort()
        pairs, deltas = pairs[keep], deltas[keep]
        print(f"Capped to {len(pairs)} pairs")

    # ---------------- identity pairs ----------------
    n_identity = int(round(args.identity_frac / max(1 - args.identity_frac, 1e-9) * len(pairs)))
    if n_identity > 0:
        pool_idx = np.flatnonzero(valid)
        ident = rng.choice(pool_idx, size=min(n_identity, len(pool_idx)), replace=False).astype(np.int32)
        pairs = np.concatenate([pairs, np.stack([ident, ident], axis=1)])
        deltas = np.concatenate([deltas, np.zeros((len(ident), labels.shape[1]), dtype=np.float32)])
        print(f"Added {len(ident)} identity pairs (delta = 0)")

    shuffle = rng.permutation(len(pairs))
    pairs, deltas = pairs[shuffle], deltas[shuffle]

    np.save(pairs_path, pairs)
    np.save(corpus_dir / "pair_deltas.npy", deltas)

    meta = {
        "n_pairs": int(len(pairs)),
        "n_identity": int(n_identity),
        "settings": vars(args),
        "delta_stats": {},
    }
    print(f"\n{'delta':<8} {'mean':>9} {'std':>9} {'p1':>9} {'p50':>9} {'p99':>9} {'|d|>1':>8}")
    print("-" * 66)
    for i, name in enumerate(PROPERTIES):
        d = deltas[:, i]
        frac_big = float((np.abs(d) > 1.0).mean())
        meta["delta_stats"][name] = {
            "mean": float(d.mean()), "std": float(d.std()),
            "p1": float(np.percentile(d, 1)), "p50": float(np.percentile(d, 50)),
            "p99": float(np.percentile(d, 99)), "frac_abs_gt_1": frac_big,
            "histogram": np.histogram(d, bins=40)[0].tolist(),
            "histogram_edges": [float(e) for e in np.histogram(d, bins=40)[1]],
        }
        print(f"{name:<8} {d.mean():>9.3f} {d.std():>9.3f} {np.percentile(d, 1):>9.3f} "
              f"{np.percentile(d, 50):>9.3f} {np.percentile(d, 99):>9.3f} {frac_big:>8.1%}")

    (corpus_dir / "pairs_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {pairs_path} {pairs.shape} and pair_deltas.npy {deltas.shape}")
    print("\nRead the |d|>1 column for logp before training. If it is near zero, every pair is")
    print("a near-copy and the conditioning has nothing to learn: lower --tanimoto-max, or")
    print("re-run with --balance-delta logp (tune --balance-cap-quantile off the histogram in")
    print("pairs_meta.json -- it discards pairs, so check how many survive).")


if __name__ == "__main__":
    main()
