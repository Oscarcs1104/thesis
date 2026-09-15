"""Block 1a -- build the MOSES pretraining corpus: SMILES -> canonical -> SELFIES -> int16 tensor.

MOSES (~1.9M drug-like molecules from ZINC) is the pretraining corpus for the
conditional generator. Chosen over a bigger raw dump because it is curated and has
published baselines (validity / uniqueness / novelty / FCD) to compare against.

What this produces, all under --out-dir:

    corpus.csv            index, smiles (canonical), inchikey
    selfies_tokens.npy    int16 [N, max_len]  -- padded with 0 (<PAD>)
    selfies_lengths.npy   int16 [N]           -- real token count per molecule
    vocab.json            token <-> id, compatible with common/selfies_vocab.py's layout
    dedup_report.json     what was removed for overlapping with the evaluation test sets
    meta.json             counts, token-length percentiles, the exact settings used

Deduplication (the part that has to survive a defense): every molecule whose InChIKey
appears in the TEST split of esol / freesolv / lipo is dropped from the corpus, because
Block 4 fine-tunes on those same datasets and any overlap would make the transfer result
indefensible. InChIKey, not raw SMILES, so a differently-written form of the same molecule
is still caught. Scaffold overlap is *reported* but not removed -- removing it would strip
whole chemotypes and is not what the literature does; the number just has to be stated.

Usage:
    python data_pipeline/moses.py                       # downloads, then caches
    python data_pipeline/moses.py --csv /path/dataset_v1.csv   # if you fetched it manually
    python data_pipeline/moses.py --limit 50000         # quick smoke run
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# MOSES ships dataset_v1.csv through Git LFS; this is the resolved media URL.
MOSES_URL = "https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv"

PAD_TOKEN = "<PAD>"
START_TOKEN = "<START>"
END_TOKEN = "<END>"
UNK_TOKEN = "<OTHER>"
SPECIAL_TOKENS = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]

# Test splits whose molecules must not appear in the pretraining corpus.
EVAL_TEST_CSVS = [
    ("esol", "data/deepchem_molnet/delaney/csv/test.csv"),
    ("freesolv", "data/deepchem_molnet/freesolv/csv/test.csv"),
    ("lipo", "data/deepchem_molnet/lipo/csv/test.csv"),
]


# --------------------------------------------------------------------------- #
# Worker functions (module level so multiprocessing can pickle them)
# --------------------------------------------------------------------------- #
def _prepare_one(smiles: str) -> Optional[Tuple[str, str, str, int]]:
    """(canonical_smiles, inchikey, selfies, n_tokens) or None if RDKit/SELFIES reject it."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    import selfies as sf

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    canonical = Chem.MolToSmiles(mol)
    try:
        key = Chem.MolToInchiKey(mol)
    except Exception:
        return None
    if not key:
        return None
    try:
        encoded = sf.encoder(canonical)
        n_tokens = len(list(sf.split_selfies(encoded)))
    except Exception:
        # A molecule SELFIES can't represent is useless to us: the decoder's
        # validity-by-construction guarantee is the whole reason we use SELFIES.
        return None
    return canonical, key, encoded, n_tokens


def _scaffold_one(smiles: str) -> str:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    from rdkit.Chem.Scaffolds import MurckoScaffold

    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)
    except Exception:
        return ""


def _tokenize_one(args: Tuple[str, Dict[str, int], int]) -> List[int]:
    """SELFIES string -> padded id row. <START>/<END> are added by the trainer, not here:
    storing them would waste two slots per molecule across 1.9M rows and force a
    re-encode if the special-token layout ever changes."""
    encoded, token_to_id, max_len = args
    import selfies as sf

    unk = token_to_id[UNK_TOKEN]
    ids = [token_to_id.get(tok, unk) for tok in sf.split_selfies(encoded)][:max_len]
    return ids + [0] * (max_len - len(ids))


# --------------------------------------------------------------------------- #
def _imap(fn, items, workers: int, chunksize: int, desc: str):
    """Parallel map with a progress line. Falls back to serial for workers <= 1."""
    total = len(items)
    start = time.time()
    out = []
    if workers <= 1:
        for i, item in enumerate(items, 1):
            out.append(fn(item))
            if i % 100_000 == 0 or i == total:
                print(f"  {desc}: {i}/{total} ({time.time() - start:.0f}s)", flush=True)
        return out
    with mp.Pool(workers) as pool:
        for i, res in enumerate(pool.imap(fn, items, chunksize=chunksize), 1):
            out.append(res)
            if i % 100_000 == 0 or i == total:
                print(f"  {desc}: {i}/{total} ({time.time() - start:.0f}s)", flush=True)
    return out


def _load_eval_inchikeys() -> Tuple[Dict[str, set], set]:
    """InChIKeys and Murcko scaffolds of every molecule in the three TEST splits."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    per_dataset: Dict[str, set] = {}
    scaffolds: set = set()
    for name, rel in EVAL_TEST_CSVS:
        path = ROOT / rel
        if not path.exists():
            print(f"  [warn] {rel} missing -- run data_pipeline/prepare_all.py first; "
                  f"skipping {name} in the overlap check")
            per_dataset[name] = set()
            continue
        keys = set()
        for smi in pd.read_csv(path)["smiles"].astype(str):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            try:
                key = Chem.MolToInchiKey(mol)
            except Exception:
                continue
            if key:
                keys.add(key)
            scaffolds.add(_scaffold_one(Chem.MolToSmiles(mol)))
        per_dataset[name] = keys
        print(f"  {name}: {len(keys)} test molecules")
    return per_dataset, scaffolds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=str, default=None, help="local dataset_v1.csv; downloads if omitted")
    p.add_argument("--out-dir", type=str, default="data/moses")
    p.add_argument("--smiles-col", type=str, default="SMILES")
    p.add_argument("--limit", type=int, default=None, help="only the first N rows (smoke test)")
    p.add_argument("--max-len", type=int, default=None,
                   help="max SELFIES tokens; default = the --length-percentile cutoff")
    p.add_argument("--length-percentile", type=float, default=99.5)
    p.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    p.add_argument("--chunksize", type=int, default=2000)
    p.add_argument("--min-token-freq", type=int, default=5,
                   help="SELFIES tokens rarer than this collapse into <OTHER>")
    p.add_argument("--force", action="store_true", help="rebuild even if the cache exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ROOT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokens_path = out_dir / "selfies_tokens.npy"
    if tokens_path.exists() and not args.force:
        meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
        print(f"Cache already built at {out_dir} ({meta['n_molecules']} molecules, "
              f"max_len={meta['max_len']}, vocab={meta['vocab_size']}). Use --force to rebuild.")
        return

    # ---------------- raw ----------------
    raw_path = out_dir / "dataset_v1.csv"
    if args.csv:
        src = Path(args.csv)
    elif raw_path.exists():
        src = raw_path
    else:
        print(f"Downloading MOSES from {MOSES_URL} ...")
        df = pd.read_csv(MOSES_URL)
        df.to_csv(raw_path, index=False)
        src = raw_path
    df = pd.read_csv(src)
    if args.smiles_col not in df.columns:
        raise SystemExit(f"column {args.smiles_col!r} not in {src} (columns: {list(df.columns)})")
    smiles = df[args.smiles_col].astype(str).tolist()
    if args.limit:
        smiles = smiles[: args.limit]
    print(f"Read {len(smiles)} SMILES from {src}")

    # ---------------- canonicalize + SELFIES ----------------
    print(f"Canonicalizing + SELFIES-encoding on {args.workers} workers...")
    prepared = _imap(_prepare_one, smiles, args.workers, args.chunksize, "prepared")
    rows = [r for r in prepared if r is not None]
    n_rejected = len(prepared) - len(rows)
    print(f"  kept {len(rows)}, rejected {n_rejected} (unparseable or not SELFIES-representable)")

    # ---------------- de-duplicate within the corpus ----------------
    seen: Dict[str, int] = {}
    unique: List[Tuple[str, str, str, int]] = []
    for row in rows:
        if row[1] in seen:
            continue
        seen[row[1]] = len(unique)
        unique.append(row)
    print(f"  {len(rows) - len(unique)} internal duplicates (same InChIKey) merged -> {len(unique)}")

    # ---------------- remove evaluation-set leakage ----------------
    print("Checking overlap with the esol/freesolv/lipo TEST splits...")
    eval_keys, eval_scaffolds = _load_eval_inchikeys()
    all_eval_keys = set().union(*eval_keys.values()) if eval_keys else set()
    removed_per_dataset = {name: 0 for name, _ in EVAL_TEST_CSVS}
    kept: List[Tuple[str, str, str, int]] = []
    for row in unique:
        hit = [name for name, keys in eval_keys.items() if row[1] in keys]
        if hit:
            for name in hit:
                removed_per_dataset[name] += 1
            continue
        kept.append(row)
    n_removed = len(unique) - len(kept)
    print(f"  removed {n_removed} molecules present in a test split: {removed_per_dataset}")

    canonical = [r[0] for r in kept]
    inchikeys = [r[1] for r in kept]
    selfies_strings = [r[2] for r in kept]
    lengths = np.array([r[3] for r in kept], dtype=np.int32)

    # Scaffold overlap is reported, not removed: dropping every shared scaffold would
    # strip whole chemotypes from a drug-like corpus, and no published pretraining
    # protocol does it. The number just has to be on the record.
    print("Computing Murcko scaffolds for the overlap report...")
    corpus_scaffolds = _imap(_scaffold_one, canonical, args.workers, args.chunksize, "scaffolds")
    shared = eval_scaffolds & set(corpus_scaffolds)
    shared.discard("")
    n_mols_shared_scaffold = sum(1 for s in corpus_scaffolds if s in shared)
    print(f"  {len(shared)} scaffolds shared with the test sets, covering "
          f"{n_mols_shared_scaffold} corpus molecules ({n_mols_shared_scaffold / max(len(canonical), 1):.1%}) -- reported, not removed")

    # ---------------- length cutoff ----------------
    pcts = {str(p): float(np.percentile(lengths, p)) for p in (50, 90, 99, 99.5, 100)}
    max_len = args.max_len or int(np.ceil(np.percentile(lengths, args.length_percentile)))
    n_truncated = int((lengths > max_len).sum())
    print(f"SELFIES token length: median={pcts['50']:.0f} p99={pcts['99']:.0f} max={pcts['100']:.0f}"
          f"  ->  max_len={max_len} ({n_truncated} molecules truncated)")

    # ---------------- vocabulary ----------------
    import selfies as sf

    counts: Counter = Counter()
    for s in selfies_strings:
        counts.update(sf.split_selfies(s))
    frequent = sorted(tok for tok, c in counts.items() if c >= args.min_token_freq)
    token_to_id = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for tok in frequent:
        token_to_id.setdefault(tok, len(token_to_id))
    rare = len(counts) - len(frequent)
    print(f"Vocabulary: {len(token_to_id)} tokens ({len(SPECIAL_TOKENS)} special, {len(frequent)} SELFIES, "
          f"{rare} rare tokens folded into {UNK_TOKEN})")
    if len(token_to_id) > np.iinfo(np.int16).max:
        raise SystemExit("vocabulary exceeds int16; widen the token array dtype")

    # ---------------- encode ----------------
    print("Encoding to int16...")
    encoded_rows = _imap(
        _tokenize_one,
        [(s, token_to_id, max_len) for s in selfies_strings],
        args.workers, args.chunksize, "encoded",
    )
    tokens = np.asarray(encoded_rows, dtype=np.int16)
    lengths_clipped = np.minimum(lengths, max_len).astype(np.int16)

    # ---------------- write ----------------
    # scaffold column is carried along so mine_pairs.py doesn't recompute 1.9M of them
    pd.DataFrame({"smiles": canonical, "inchikey": inchikeys, "scaffold": corpus_scaffolds}).to_csv(
        out_dir / "corpus.csv", index_label="index"
    )
    np.save(tokens_path, tokens)
    np.save(out_dir / "selfies_lengths.npy", lengths_clipped)
    (out_dir / "vocab.json").write_text(json.dumps({
        "token_to_id": token_to_id,
        "id_to_token": {str(i): t for t, i in token_to_id.items()},
        "pad_idx": token_to_id[PAD_TOKEN],
        "start_idx": token_to_id[START_TOKEN],
        "end_idx": token_to_id[END_TOKEN],
        "unk_idx": token_to_id[UNK_TOKEN],
    }, indent=2), encoding="utf-8")
    (out_dir / "dedup_report.json").write_text(json.dumps({
        "removed_exact_inchikey": {"total": n_removed, "per_dataset": removed_per_dataset},
        "eval_test_molecules": {k: len(v) for k, v in eval_keys.items()},
        "eval_test_molecules_total_unique": len(all_eval_keys),
        "scaffold_overlap": {
            "n_shared_scaffolds": len(shared),
            "n_corpus_molecules_with_shared_scaffold": n_mols_shared_scaffold,
            "fraction_of_corpus": n_mols_shared_scaffold / max(len(canonical), 1),
            "policy": "reported, not removed",
        },
    }, indent=2), encoding="utf-8")
    (out_dir / "meta.json").write_text(json.dumps({
        "source": str(src),
        "n_raw": len(smiles),
        "n_rejected": n_rejected,
        "n_internal_duplicates": len(rows) - len(unique),
        "n_removed_eval_overlap": n_removed,
        "n_molecules": len(canonical),
        "max_len": max_len,
        "n_truncated": n_truncated,
        "length_percentiles": pcts,
        "vocab_size": len(token_to_id),
        "min_token_freq": args.min_token_freq,
    }, indent=2), encoding="utf-8")

    print(f"\nWrote {len(canonical)} molecules to {out_dir}")
    print(f"  tokens {tokens.shape} {tokens.dtype} ({tokens.nbytes / 1e6:.0f} MB)")
    print("Next: python data_pipeline/rdkit_labels.py --corpus-dir", args.out_dir)


if __name__ == "__main__":
    main()
