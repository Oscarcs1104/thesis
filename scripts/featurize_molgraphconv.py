"""Dump dc.feat.MolGraphConvFeaturizer features to disk, once, from a throwaway env.

    # in an environment that has deepchem, and needs nothing else:
    python scripts/featurize_molgraphconv.py

DeepChem is needed for exactly one thing here -- the node features the reference MoLA
benchmark uses -- and it drags TensorFlow in eagerly along with its own pins on numpy and
protobuf. Installing that beside a working torch/PyG environment risks downgrading numpy
under compiled wheels, which is a bad trade for one featurizer. So it runs alone, writes
an .npz, and the training environment never imports it.

Writes data/deepchem_molnet/<dir>/csv/molgraphconv.npz, holding the pool in the order the
three CSVs concatenate, flat node-feature and edge arrays with offsets, and the indices
the featurizer accepted. That last one matters: the reference drops what the featurizer
refuses BEFORE partitioning, so the pool is 1127 ESOL molecules rather than 1128 and every
index after the dropped one shifts. Storing the kept indices keeps that reproducible here.

Re-run it whenever the split CSVs change: the file records a digest of what it was built
from and the loader refuses a stale one.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Kept in step with crossmodal_model/train/core.py's DATASETS, but spelled out so this
# script imports nothing from a package that needs torch.
DATASETS = {
    "esol": ("delaney", "y"),
    "freesolv": ("freesolv", "y"),
    "lipo": ("lipo", "y"),
}


def pool_digest(smiles, targets) -> str:
    h = hashlib.sha256()
    h.update(str(len(smiles)).encode())
    for s, t in zip(smiles, targets):
        h.update(str(len(s)).encode())
        h.update(s.encode("utf-8"))
        h.update(f"{t:.10g}".encode())
    return h.hexdigest()


def main() -> None:
    try:
        import deepchem as dc
    except Exception as exc:  # noqa: BLE001
        print(f"Could not import DeepChem: {type(exc).__name__}: {exc}")
        print("  pip install deepchem tensorflow-cpu")
        print("This script is meant to run in an environment of its own -- it needs")
        print("numpy, pandas and deepchem, and nothing from the training stack.")
        raise SystemExit(2)

    for name, (subdir, target_col) in DATASETS.items():
        csv_dir = ROOT / "data" / "deepchem_molnet" / subdir / "csv"
        if not (csv_dir / "train.csv").exists():
            print(f"[{name}] no split CSVs at {csv_dir}; run data_pipeline/prepare_all.py")
            continue

        pool = pd.concat([pd.read_csv(csv_dir / f"{s}.csv") for s in ("train", "valid", "test")],
                         ignore_index=True)
        smiles = pool["smiles"].astype(str).tolist()
        targets = pool[target_col].astype(float).tolist()

        feats = dc.feat.MolGraphConvFeaturizer().featurize(smiles)
        keep = [i for i, f in enumerate(feats)
                if hasattr(f, "node_features") and hasattr(f, "edge_index")]
        if len(keep) != len(smiles):
            print(f"[{name}] featurizer dropped {len(smiles) - len(keep)}/{len(smiles)}")

        nodes = [np.asarray(feats[i].node_features, dtype=np.float32) for i in keep]
        edges = [np.asarray(feats[i].edge_index, dtype=np.int32) for i in keep]
        node_ptr = np.concatenate([[0], np.cumsum([n.shape[0] for n in nodes])]).astype(np.int64)
        edge_ptr = np.concatenate([[0], np.cumsum([e.shape[1] for e in edges])]).astype(np.int64)

        out = csv_dir / "molgraphconv.npz"
        np.savez_compressed(
            out,
            x=np.concatenate(nodes, axis=0),
            edge_index=np.concatenate(edges, axis=1),
            node_ptr=node_ptr,
            edge_ptr=edge_ptr,
            keep=np.asarray(keep, dtype=np.int64),
            # Fixed-width unicode, not dtype=object: an object array in an .npz can only
            # be read back with allow_pickle=True, and nothing here needs that.
            smiles=np.asarray([smiles[i] for i in keep]),
            y=np.asarray([targets[i] for i in keep], dtype=np.float64),
            digest=pool_digest(smiles, targets),
            n_pool=len(smiles),
        )
        mb = out.stat().st_size / 1e6
        print(f"[{name}] {len(keep)} molecules, {node_ptr[-1]:,} atoms, "
              f"{nodes[0].shape[1]} node features -> {out} ({mb:.1f} MB)")

    print("\nCopy nothing: the .npz sits next to the CSVs it came from. The training")
    print("environment reads it without importing DeepChem.")


if __name__ == "__main__":
    main()
