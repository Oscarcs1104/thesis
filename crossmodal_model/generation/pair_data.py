"""Dataset for lead optimization: (M_a, delta property) -> M_b.

Each of the ~13M mined pairs names two molecules out of ~1.94M, so every molecule is a
source or target roughly a dozen times. Featurizing per pair would run RDKit ~13M times
an epoch; holding 1.94M PyG Data objects would cost tens of gigabytes in Python object
overhead alone. So molecules are featurized once into FLAT arrays with offsets --
around 1.5 GB for the whole corpus -- and a Data object is assembled per item from
slices. The cache is written to disk, so it is built once ever.

Splitting is by SCAFFOLD. Pairs were mined inside scaffold buckets
(data_pipeline/mine_pairs.py), so a scaffold-level split keeps every pair intact
automatically while guaranteeing no molecule appears on two sides. Splitting by pair
would put the same molecule in train and test as source and target, and the held-out
numbers would measure memorization.

Reads the cached corpus from data_pipeline/{moses,rdkit_labels,mine_pairs}.py:

    corpus.csv          smiles, scaffold
    labels.npy          [N, 4] logP / TPSA / QED / MW
    selfies_tokens.npy  [N, L] int16 -- the decoder target
    vocab.json          the SELFIES vocabulary
    pairs.npy           [P, 2] int32 source, target
    pair_deltas.npy     [P, 4] float32 labels[target] - labels[source]
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset as GeomDataset

from common.property_bins import PropertyBinner
from data_pipeline.features import ATOM_FEATURE_DIMS, BOND_FEATURE_DIMS
from data_pipeline.rdkit_labels import PROPERTIES

CACHE_NAME = "molecule_graph_cache.npz"


def build_char_vocab(smiles: Sequence[str]) -> Dict[str, int]:
    """Character vocabulary for the encoder's SMILES branch; index 0 is padding."""
    chars = sorted({c for s in smiles for c in s})
    vocab = {c: i + 1 for i, c in enumerate(chars)}
    vocab["<pad>"] = 0
    return vocab


_WORKER: dict = {}


def _init_worker(char_vocab: Dict[str, int], max_sm_len: int) -> None:
    _WORKER["char_vocab"] = char_vocab
    _WORKER["max_sm_len"] = max_sm_len
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")


def _featurize_one(smiles: str):
    """(x, edge_index, edge_attr, sm) as small numpy arrays, or None if RDKit refuses."""
    from data_pipeline.convert_smiles_to_pyg import smiles_to_data

    d = smiles_to_data(smiles)
    if d is None:
        return None
    vocab, max_len = _WORKER["char_vocab"], _WORKER["max_sm_len"]
    sm = np.zeros(max_len, dtype=np.int16)
    for i, c in enumerate(smiles[:max_len]):
        sm[i] = vocab.get(c, 0)
    return (
        d.x.numpy().astype(np.int8),               # every OGB column fits: max index 118
        d.edge_index.numpy().astype(np.int16),     # local atom indices, molecules are small
        d.edge_attr.numpy().astype(np.int8),
        sm,
    )


class MoleculeGraphCache:
    """Flat per-atom / per-edge arrays plus offsets. Assembles a Data object on demand."""

    def __init__(self, arrays: Dict[str, np.ndarray], char_vocab: Dict[str, int]) -> None:
        self.x = arrays["x"]
        self.edge_index = arrays["edge_index"]
        self.edge_attr = arrays["edge_attr"]
        self.sm = arrays["sm"]
        self.atom_ptr = arrays["atom_ptr"]
        self.edge_ptr = arrays["edge_ptr"]
        self.valid = arrays["valid"].astype(bool)
        self.char_vocab = char_vocab

    def __len__(self) -> int:
        return len(self.atom_ptr) - 1

    def get(self, i: int) -> Data:
        a0, a1 = int(self.atom_ptr[i]), int(self.atom_ptr[i + 1])
        e0, e1 = int(self.edge_ptr[i]), int(self.edge_ptr[i + 1])
        return Data(
            x=torch.from_numpy(self.x[a0:a1].astype(np.int64)),
            edge_index=torch.from_numpy(self.edge_index[:, e0:e1].astype(np.int64)),
            edge_attr=torch.from_numpy(self.edge_attr[e0:e1].astype(np.int64)),
            sm=torch.from_numpy(self.sm[i].astype(np.int64)).unsqueeze(0),
        )

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path, x=self.x, edge_index=self.edge_index, edge_attr=self.edge_attr,
            sm=self.sm, atom_ptr=self.atom_ptr, edge_ptr=self.edge_ptr,
            valid=self.valid, char_vocab=json.dumps(self.char_vocab),
        )

    @classmethod
    def load(cls, path: Path) -> "MoleculeGraphCache":
        z = np.load(path, allow_pickle=False)
        arrays = {k: z[k] for k in ("x", "edge_index", "edge_attr", "sm", "atom_ptr", "edge_ptr", "valid")}
        return cls(arrays, json.loads(str(z["char_vocab"])))

    @classmethod
    def build(cls, smiles: Sequence[str], char_vocab: Dict[str, int], max_sm_len: int = 100,
              workers: int = 8, chunksize: int = 2000) -> "MoleculeGraphCache":
        n = len(smiles)
        print(f"Featurizing {n:,} molecules on {workers} workers...")
        start = time.time()
        results: List[Optional[tuple]] = []
        with mp.Pool(workers, initializer=_init_worker, initargs=(char_vocab, max_sm_len)) as pool:
            for i, r in enumerate(pool.imap(_featurize_one, smiles, chunksize=chunksize), 1):
                results.append(r)
                if i % 250_000 == 0:
                    print(f"  {i:,}/{n:,} ({time.time() - start:.0f}s)", flush=True)

        valid = np.array([r is not None for r in results])
        n_atoms = np.array([0 if r is None else r[0].shape[0] for r in results], dtype=np.int64)
        n_edges = np.array([0 if r is None else r[1].shape[1] for r in results], dtype=np.int64)
        atom_ptr = np.concatenate([[0], np.cumsum(n_atoms)])
        edge_ptr = np.concatenate([[0], np.cumsum(n_edges)])

        x = np.zeros((int(atom_ptr[-1]), len(ATOM_FEATURE_DIMS)), dtype=np.int8)
        edge_index = np.zeros((2, int(edge_ptr[-1])), dtype=np.int16)
        edge_attr = np.zeros((int(edge_ptr[-1]), len(BOND_FEATURE_DIMS)), dtype=np.int8)
        sm = np.zeros((n, max_sm_len), dtype=np.int16)
        for i, r in enumerate(results):
            if r is None:
                continue
            x[atom_ptr[i]:atom_ptr[i + 1]] = r[0]
            edge_index[:, edge_ptr[i]:edge_ptr[i + 1]] = r[1]
            edge_attr[edge_ptr[i]:edge_ptr[i + 1]] = r[2]
            sm[i] = r[3]

        total_mb = sum(a.nbytes for a in (x, edge_index, edge_attr, sm)) / 1e6
        print(f"  cache: {int(valid.sum()):,} molecules, {total_mb:.0f} MB in RAM "
              f"({time.time() - start:.0f}s)")
        return cls({"x": x, "edge_index": edge_index, "edge_attr": edge_attr, "sm": sm,
                    "atom_ptr": atom_ptr, "edge_ptr": edge_ptr, "valid": valid}, char_vocab)


class PairDataset(GeomDataset):
    """One item = (M_a graph + chars, delta bins, SELFIES tokens of M_b)."""

    def __init__(self, cache: MoleculeGraphCache, pairs: np.ndarray,
                 cond_bins: np.ndarray, targets: np.ndarray) -> None:
        super().__init__()
        self.cache = cache
        self.pairs = pairs
        self.cond_bins = torch.from_numpy(cond_bins)
        self.targets = targets

    def len(self) -> int:
        return len(self.pairs)

    def get(self, idx: int) -> Data:
        src, tgt = int(self.pairs[idx, 0]), int(self.pairs[idx, 1])
        d = self.cache.get(src)
        d.cond = self.cond_bins[idx].unsqueeze(0)
        d.tgt = torch.from_numpy(self.targets[tgt].astype(np.int64)).unsqueeze(0)
        return d


def build_pair_datasets(
    corpus_dir: Path,
    num_bins: int = 20,
    max_sm_len: int = 100,
    val_frac: float = 0.02,
    test_frac: float = 0.02,
    seed: int = 2025,
    workers: int = 8,
    max_pairs: Optional[int] = None,
    rebuild_cache: bool = False,
):
    """Returns (train, val, test, cache, binners, vocab)."""
    corpus_dir = Path(corpus_dir)
    corpus = pd.read_csv(corpus_dir / "corpus.csv")
    labels = np.load(corpus_dir / "labels.npy")
    targets = np.load(corpus_dir / "selfies_tokens.npy")
    pairs = np.load(corpus_dir / "pairs.npy")
    deltas = np.load(corpus_dir / "pair_deltas.npy")
    vocab = json.loads((corpus_dir / "vocab.json").read_text(encoding="utf-8"))
    vocab["id_to_token"] = {int(k): v for k, v in vocab["id_to_token"].items()}
    smiles = corpus["smiles"].astype(str).tolist()

    cache_path = corpus_dir / CACHE_NAME
    if cache_path.exists() and not rebuild_cache:
        cache = MoleculeGraphCache.load(cache_path)
        print(f"Loaded graph cache from {cache_path} ({len(cache):,} molecules)")
    else:
        cache = MoleculeGraphCache.build(smiles, build_char_vocab(smiles), max_sm_len, workers)
        cache.save(cache_path)
        print(f"Saved graph cache to {cache_path}")

    if max_pairs and len(pairs) > max_pairs:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.permutation(len(pairs))[:max_pairs])
        pairs, deltas = pairs[keep], deltas[keep]
        print(f"Subsampled to {len(pairs):,} pairs")

    # Pairs were mined inside scaffold buckets, so assigning whole scaffolds to a split
    # keeps every pair on one side while guaranteeing no molecule appears on two.
    scaffolds = corpus["scaffold"].fillna("").astype(str).to_numpy()
    uniq, scaffold_id = np.unique(scaffolds, return_inverse=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    n_val, n_test = int(len(uniq) * val_frac), int(len(uniq) * test_frac)
    split_of_scaffold = np.zeros(len(uniq), dtype=np.int8)
    split_of_scaffold[order[:n_val]] = 1
    split_of_scaffold[order[n_val:n_val + n_test]] = 2

    src_split = split_of_scaffold[scaffold_id[pairs[:, 0]]]
    tgt_split = split_of_scaffold[scaffold_id[pairs[:, 1]]]
    intact = src_split == tgt_split
    if not intact.all():
        # Only possible for pairs that somehow cross buckets; drop them rather than
        # letting a training molecule leak into the test side.
        print(f"  dropping {int((~intact).sum()):,} pairs whose ends fall in different splits")
        pairs, deltas, src_split = pairs[intact], deltas[intact], src_split[intact]

    usable = cache.valid[pairs[:, 0]] & cache.valid[pairs[:, 1]]
    if not usable.all():
        print(f"  dropping {int((~usable).sum()):,} pairs with an unfeaturizable molecule")
        pairs, deltas, src_split = pairs[usable], deltas[usable], src_split[usable]

    # Bin edges are fit on TRAIN deltas only; fitting them on everything would leak the
    # test split's delta distribution into what the model is asked to produce.
    train_mask = src_split == 0
    binners = {
        name: PropertyBinner.fit(deltas[train_mask, p], num_bins=num_bins, name=name)
        for p, name in enumerate(PROPERTIES)
    }

    def _make(mask: np.ndarray) -> PairDataset:
        d = deltas[mask]
        bins = np.stack([binners[n].to_bins(d[:, p]) for p, n in enumerate(PROPERTIES)], axis=1)
        return PairDataset(cache, pairs[mask], bins.astype(np.int64), targets)

    out = {name: _make(src_split == code) for name, code in (("train", 0), ("val", 1), ("test", 2))}
    print(f"Split by scaffold: train={len(out['train']):,} val={len(out['val']):,} "
          f"test={len(out['test']):,} pairs over {len(uniq):,} scaffolds")
    return out["train"], out["val"], out["test"], cache, binners, vocab


__all__ = ["MoleculeGraphCache", "PairDataset", "build_pair_datasets", "build_char_vocab"]
