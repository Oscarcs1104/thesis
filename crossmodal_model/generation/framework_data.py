"""Dataset for framework-conditioned generation: (generic framework, property) -> molecule.

The encoder input is a generic framework, and there are only ~70.9k distinct frameworks
for 1.94M molecules. So every encoder input is featurized ONCE at startup and indexed
afterwards -- RDKit never runs inside the training loop, and no multi-gigabyte graph
cache is needed. Molecules sharing a framework share the same tensors; they are never
mutated, so sharing is safe and costs nothing.

Reads the cached corpus built by data_pipeline/{moses,rdkit_labels,report_corpus}.py:

    corpus.csv           smiles, scaffold
    frameworks.csv       the generic framework per molecule  (report_corpus.py --frameworks)
    labels.npy           [N, 4] float32 logP / TPSA / QED / MW
    selfies_tokens.npy   [N, L] int16 -- the DECODER TARGET, already tokenized
    vocab.json           the SELFIES vocabulary

Split is by FRAMEWORK, not by molecule: molecules sharing a framework would otherwise
land on both sides, and the model could score well on validation by reproducing a
framework it had already been trained to fill in. Splitting on the constraint is the
only way the held-out numbers mean "generalizes to unseen topologies".
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset as GeomDataset

from common.property_bins import PropertyBinner
from data_pipeline.convert_smiles_to_pyg import smiles_to_data
from data_pipeline.rdkit_labels import PROPERTIES


def build_char_vocab(texts: Sequence[str]) -> Dict[str, int]:
    """Character vocabulary for the encoder's SMILES branch (index 0 is padding).

    Built over the FRAMEWORK strings, not over molecules: the framework is what the
    encoder ever sees, and its alphabet is tiny (C, ring digits, parentheses).
    """
    chars = sorted({c for t in texts for c in t})
    vocab = {c: i + 1 for i, c in enumerate(chars)}
    vocab["<pad>"] = 0
    return vocab


def _encode_chars(text: str, vocab: Dict[str, int], max_len: int) -> torch.Tensor:
    ids = [vocab.get(c, 0) for c in text[:max_len]]
    ids.extend([0] * (max_len - len(ids)))
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


class FrameworkCache:
    """Featurizes every distinct framework once: PyG graph + SMILES character indices."""

    def __init__(self, frameworks: Sequence[str], max_sm_len: int = 100,
                 char_vocab: Optional[Dict[str, int]] = None) -> None:
        self.unique: List[str] = sorted({f for f in frameworks if f})
        self.index: Dict[str, int] = {f: i for i, f in enumerate(self.unique)}
        self.char_vocab = char_vocab or build_char_vocab(self.unique)
        self.max_sm_len = max_sm_len

        self.graphs: List[Optional[Data]] = []
        self.failed: List[str] = []
        for fw in self.unique:
            d = smiles_to_data(fw)
            if d is None:
                self.graphs.append(None)
                self.failed.append(fw)
                continue
            d.sm = _encode_chars(fw, self.char_vocab, max_sm_len)
            self.graphs.append(d)

    def __len__(self) -> int:
        return len(self.unique)

    def usable(self, fw: str) -> bool:
        i = self.index.get(fw, -1)
        return i >= 0 and self.graphs[i] is not None


class FrameworkConditionalDataset(GeomDataset):
    """One item = (framework graph + chars, property bins, target SELFIES tokens)."""

    def __init__(self, cache: FrameworkCache, framework_ids: np.ndarray,
                 cond_bins: np.ndarray, targets: np.ndarray) -> None:
        super().__init__()
        self.cache = cache
        self.framework_ids = framework_ids            # [n] index into cache.unique
        self.cond_bins = torch.from_numpy(cond_bins)  # [n, P] int64
        self.targets = torch.from_numpy(targets)      # [n, L] int16

    def len(self) -> int:
        return len(self.framework_ids)

    def get(self, idx: int) -> Data:
        base = self.cache.graphs[int(self.framework_ids[idx])]
        # Tensors are shared with the cache rather than cloned: nothing mutates them, and
        # copying a graph 27 times per epoch for no reason is the kind of overhead that
        # quietly doubles epoch time.
        return Data(
            x=base.x, edge_index=base.edge_index, edge_attr=base.edge_attr, sm=base.sm,
            cond=self.cond_bins[idx].unsqueeze(0),
            tgt=self.targets[idx].long().unsqueeze(0),
        )


def load_corpus(corpus_dir: Path) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    corpus_dir = Path(corpus_dir)
    fw_path = corpus_dir / "frameworks.csv"
    if not fw_path.exists():
        raise SystemExit(
            f"{fw_path} is missing. Build it with:\n"
            f"  python data_pipeline/report_corpus.py --corpus-dir {corpus_dir} --frameworks"
        )
    corpus = pd.read_csv(corpus_dir / "corpus.csv")
    corpus["framework"] = pd.read_csv(fw_path)["framework"].fillna("").astype(str)
    labels = np.load(corpus_dir / "labels.npy")
    tokens = np.load(corpus_dir / "selfies_tokens.npy")
    vocab = json.loads((corpus_dir / "vocab.json").read_text(encoding="utf-8"))
    vocab["id_to_token"] = {int(k): v for k, v in vocab["id_to_token"].items()}
    return corpus, labels, tokens, vocab


def build_datasets(
    corpus_dir: Path,
    num_bins: int = 20,
    max_sm_len: int = 100,
    val_frac: float = 0.02,
    test_frac: float = 0.02,
    seed: int = 2025,
    limit: Optional[int] = None,
):
    """Returns (train, val, test, cache, binners, vocab, char_vocab)."""
    corpus, labels, tokens, vocab = load_corpus(corpus_dir)
    if limit:
        corpus, labels, tokens = corpus.iloc[:limit], labels[:limit], tokens[:limit]

    frameworks = corpus["framework"].astype(str).to_numpy()
    keep = np.isfinite(labels).all(axis=1) & (frameworks != "")
    n_dropped_label = int((~np.isfinite(labels).all(axis=1)).sum())
    n_acyclic = int((frameworks == "").sum())
    print(f"Corpus {len(corpus):,} molecules | dropping {n_dropped_label:,} unlabelled, "
          f"{n_acyclic:,} acyclic (no framework)")

    cache = FrameworkCache(frameworks[keep], max_sm_len=max_sm_len)
    print(f"Featurized {len(cache):,} distinct frameworks"
          + (f" ({len(cache.failed)} unparseable)" if cache.failed else ""))

    usable = keep.copy()
    fw_idx = np.full(len(corpus), -1, dtype=np.int64)
    for i in np.flatnonzero(keep):
        j = cache.index.get(frameworks[i], -1)
        if j < 0 or cache.graphs[j] is None:
            usable[i] = False
        else:
            fw_idx[i] = j
    print(f"Usable molecules: {int(usable.sum()):,}")

    # Split on frameworks, not molecules -- see the module docstring.
    rng = np.random.default_rng(seed)
    uniq = np.unique(fw_idx[usable])
    rng.shuffle(uniq)
    n_val, n_test = int(len(uniq) * val_frac), int(len(uniq) * test_frac)
    val_fw, test_fw = set(uniq[:n_val].tolist()), set(uniq[n_val:n_val + n_test].tolist())

    binners = {}
    splits: Dict[str, np.ndarray] = {}
    idx_all = np.flatnonzero(usable)
    where = np.array([2 if f in test_fw else 1 if f in val_fw else 0 for f in fw_idx[idx_all]])
    for name, code in (("train", 0), ("val", 1), ("test", 2)):
        splits[name] = idx_all[where == code]

    # Bin edges are fit on TRAIN only: fitting them on everything would leak the test
    # split's property distribution into what the model is asked to generate.
    for p, name in enumerate(PROPERTIES):
        binners[name] = PropertyBinner.fit(labels[splits["train"], p], num_bins=num_bins, name=name)

    def _make(idx: np.ndarray) -> FrameworkConditionalDataset:
        bins = np.stack([binners[n].to_bins(labels[idx, p]) for p, n in enumerate(PROPERTIES)], axis=1)
        return FrameworkConditionalDataset(cache, fw_idx[idx], bins.astype(np.int64), tokens[idx])

    out = {k: _make(v) for k, v in splits.items()}
    print(f"Split by framework: train={len(out['train']):,} val={len(out['val']):,} "
          f"test={len(out['test']):,} over {len(uniq):,} frameworks")
    return out["train"], out["val"], out["test"], cache, binners, vocab, cache.char_vocab


__all__ = ["FrameworkCache", "FrameworkConditionalDataset", "build_datasets", "build_char_vocab"]
