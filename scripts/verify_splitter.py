"""Check deepchem_random_split_subsets against the real dc.splits.RandomSplitter.

    python scripts/verify_splitter.py

data_pipeline/splitters.py reproduces DeepChem's splitter from its documented behaviour
so results can be compared molecule-for-molecule against work that uses it, without
importing DeepChem and dragging TensorFlow into the critical path. A reimplementation
cannot check itself, and the docstring says to verify it once. This is that once.

Comparing dumped split files does not do it: two pipelines that disagree about how many
molecules the pool holds produce different partitions no matter how the splitter behaves,
which is exactly what happened -- a reference ESOL test set of 113 against 112 here,
because one pool was the raw 1128 (1127 after the featurizer drops one) and the other the
1117 that remain once duplicate InChIKeys are merged. So this fixes the pool and varies
only the splitter, which is the comparison that answers the question.

Run it where DeepChem is installed. If it is not, the script says so and exits without
pretending to have verified anything.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def main() -> None:
    try:
        import deepchem as dc
        import deepchem.splits  # noqa: F401  -- the submodule actually used
    except Exception as exc:
        # Not `except ImportError` with a generic message: DeepChem imports a stack of
        # optional backends, and "not installed" and "installed but its own import chain
        # fails" need different fixes. Swallowing the cause sent someone to reinstall a
        # package that was already there.
        import traceback

        print("Could not import DeepChem, so nothing was verified.")
        print(f"  {type(exc).__name__}: {exc}\n")
        traceback.print_exc()
        print("\nIf it is genuinely absent:  pip install deepchem")
        print("If it is present but its import chain breaks, the missing piece is named")
        print("above -- often a backend DeepChem imports eagerly.")
        print("\nUntil this runs, say the partitions follow the same procedure -- not")
        print("that they are identical.")
        raise SystemExit(2)

    import numpy as np

    from data_pipeline.splitters import deepchem_random_split_subsets

    # Sizes chosen to exercise the cutoff arithmetic where it rounds: 1117 is ESOL after
    # merging duplicate InChIKeys, 1127 is the pool the reference split came from, and
    # the rest are small enough that an off-by-one in a cutoff changes a partition size.
    sizes = [10, 13, 100, 642, 1117, 1127, 1128, 4200]
    seeds = [2025, 2026, 2027]

    failures = []
    for n in sizes:
        pool = dc.data.NumpyDataset(
            X=np.arange(n).reshape(-1, 1),
            y=np.zeros((n, 1)),
            ids=np.array([f"m{i}" for i in range(n)]),
        )
        for seed in seeds:
            tr_ref, va_ref, te_ref = dc.splits.RandomSplitter().train_valid_test_split(
                pool, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=seed
            )
            ref = [[int(i) for i in d.X.reshape(-1)] for d in (tr_ref, va_ref, te_ref)]
            ours = [list(s.indices) for s in
                    deepchem_random_split_subsets(range(n), 0.8, 0.1, 0.1, seed)]

            sizes_ref, sizes_ours = [len(p) for p in ref], [len(p) for p in ours]
            if sizes_ref != sizes_ours:
                failures.append(f"n={n} seed={seed}: sizes {sizes_ours} vs DeepChem {sizes_ref}")
            elif ref != ours:
                # Same sizes, different members: the cutoffs agree and the permutation
                # does not, which is the half that decides which molecule goes where.
                shared = len(set(ref[2]) & set(ours[2]))
                failures.append(f"n={n} seed={seed}: same sizes, different members "
                                f"(test sets share {shared}/{len(ref[2])})")
            else:
                print(f"  n={n:>5} seed={seed}  train/valid/test = "
                      f"{'/'.join(map(str, sizes_ours))}  identical")

    print()
    if failures:
        print(f"{len(failures)} MISMATCH(ES):")
        for f in failures:
            print(f"  {f}")
        print("\ndeepchem_random_split_subsets does NOT reproduce dc.splits.RandomSplitter.")
        print("Do not claim identical partitions; fix the function or drop the claim.")
        raise SystemExit(1)

    print(f"All {len(sizes) * len(seeds)} combinations identical, indices and order.")
    print("deepchem_random_split_subsets reproduces dc.splits.RandomSplitter exactly.")


if __name__ == "__main__":
    main()
