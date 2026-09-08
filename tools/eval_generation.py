"""Generative evaluation harness (plan Days 9 + 10).

Given a conditional-generator checkpoint (training/train_generator.py --mode
finetune), for every target property bin it samples molecules and reports:

  validity       -- SELFIES-level (should be ~100%) AND SMILES-level (the real number)
  uniqueness     -- distinct canonical SMILES / valid
  novelty        -- fraction of unique valid not in the reference set (ZINC) and,
                    separately, not in the dataset train split
  internal div.  -- 1 - mean pairwise Tanimoto (on a capped random subset)
  scaffold div.  -- unique Murcko scaffolds / valid, + normalized Shannon entropy
  FCD            -- Frechet ChemNet Distance vs a reference sample (needs fcd_torch)
  MAD            -- mean |predicted_property - bin_center| over unique valid mols,
                    using --predictor-checkpoint; plus the in-bin rate

Writes a JSON summary and prints a table.

    python tools/eval_generation.py \
        --generator-checkpoint checkpoints/gen_esol_finetune.pt \
        --predictor-checkpoint checkpoints/graph+lang_delaney_s2025.pt \
        --reference-csv data/zinc15_250K.csv \
        --train-csv data/deepchem_molnet/delaney/csv/train.csv \
        --num-samples 10000 --out results/generation_esol.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Optional

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import selfies as sf
from rdkit import Chem, RDLogger

from data_pipeline.pseudo_label_zinc import _load_predictor
from data_pipeline.features import smiles_to_data
from model.conditional_generator import ConditionalSmilesGenerator, PropertyBinner
from tools.mol_metrics import (
    mean_pairwise_tanimoto,
    mols_from_smiles,
    morgan_fingerprints,
    murcko_scaffold_smiles,
    normalized_shannon_entropy,
)

RDLogger.DisableLog("rdApp.*")


def _canon(smi: str) -> Optional[str]:
    m = Chem.MolFromSmiles(smi) if smi else None
    return Chem.MolToSmiles(m) if m is not None else None


def _read_smiles(path: str, limit: Optional[int] = None) -> List[str]:
    import csv

    out: List[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        col = next((c for c in fields if c and c.lower() in {"smiles", "smile", "canonical_smiles"}), fields[0])
        for row in reader:
            s = (row.get(col) or "").strip()
            if s:
                out.append(s)
            if limit and len(out) >= limit:
                break
    return out


def _load_generator(path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    vocab = ckpt["vocab"]
    a = ckpt["args"]
    model = ConditionalSmilesGenerator(
        hidden_dim=a["hidden_dim"], vocab_size=len(vocab["token_to_id"]),
        pad_idx=vocab["pad_idx"], start_idx=vocab["start_idx"], end_idx=vocab["end_idx"],
        num_property_bins=a["num_bins"], decoder_layers=a["decoder_layers"],
        decoder_heads=a["decoder_heads"], max_len=max(a["max_len"], 128), dropout=a.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(ckpt["generator_state"], strict=True)
    model.eval()
    binner = PropertyBinner.from_state_dict(ckpt["binner"]) if ckpt.get("binner") else None
    return model, vocab, binner, ckpt.get("property_name", "property")


def _predict_properties(model, mean, scale, smiles: List[str], device: str, batch_size: int = 256) -> List[float]:
    from torch_geometric.loader import DataLoader as GeomDataLoader

    graphs, keep = [], []
    for i, s in enumerate(smiles):
        g = smiles_to_data(s)
        if g is not None:
            graphs.append(g)
            keep.append(i)
    preds = [float("nan")] * len(smiles)
    if not graphs:
        return preds
    vals: List[float] = []
    with torch.no_grad():
        for batch in GeomDataLoader(graphs, batch_size=batch_size, shuffle=False):
            batch = batch.to(device)
            out = model(batch).view(-1).cpu()
            vals.extend((out * scale + mean).tolist())
    for i, v in zip(keep, vals):
        preds[i] = v
    return preds


def _fcd(generated: List[str], reference: List[str]) -> Optional[float]:
    try:
        from fcd_torch import FCD
    except Exception:
        return None
    fcd = FCD(device="cuda" if torch.cuda.is_available() else "cpu", n_jobs=1)
    return float(fcd(reference, generated))


def evaluate_bin(gen, vocab, binner, bin_idx, args, reference_canon, train_canon, predictor):
    raw_selfies = gen.sample(
        bin_idx, vocab["id_to_token"], args.per_bin, max_len=args.max_len,
        temperature=args.temperature, as_selfies=True,
    )
    n = len(raw_selfies)

    smiles, selfies_ok = [], 0
    for tok in raw_selfies:
        dec = ""
        if tok:
            try:
                dec = sf.decoder(tok)
            except Exception:
                dec = ""
        if dec:
            selfies_ok += 1
        smiles.append(dec)

    canon = [_canon(s) for s in smiles]
    valid = [c for c in canon if c]
    unique = sorted(set(valid))

    res = {
        "bin": bin_idx,
        "bin_center": binner.bin_center(bin_idx) if binner else None,
        "bin_edges": list(binner.bin_edges(bin_idx)) if binner else None,
        "n_sampled": n,
        "validity_selfies": selfies_ok / max(n, 1),
        "validity_smiles": len(valid) / max(n, 1),
        "uniqueness": len(unique) / max(len(valid), 1),
    }
    if reference_canon is not None:
        res["novelty_vs_reference"] = sum(u not in reference_canon for u in unique) / max(len(unique), 1)
    if train_canon is not None:
        res["novelty_vs_train"] = sum(u not in train_canon for u in unique) / max(len(unique), 1)

    subset = unique if len(unique) <= args.diversity_cap else random.sample(unique, args.diversity_cap)
    mols, _ = mols_from_smiles(subset)
    if len(mols) >= 2:
        fps = morgan_fingerprints(mols)
        mean_sim = mean_pairwise_tanimoto(fps)
        res["internal_diversity"] = (1.0 - mean_sim) if mean_sim is not None else None
        scaffolds = [murcko_scaffold_smiles(m) for m in mols]
        res["unique_scaffolds_frac"] = len(set(scaffolds)) / len(mols)
        res["scaffold_entropy_norm"] = normalized_shannon_entropy(scaffolds)

    if args.reference_sample:
        res["fcd"] = _fcd(subset, args.reference_sample)

    if predictor is not None and unique:
        model, mean, scale = predictor
        preds = _predict_properties(model, mean, scale, unique, args.device)
        finite = [p for p in preds if p == p]
        if finite and binner:
            center = binner.bin_center(bin_idx)
            lo, hi = binner.bin_edges(bin_idx)
            res["predicted_mean"] = sum(finite) / len(finite)
            res["mad_vs_bin_center"] = sum(abs(p - center) for p in finite) / len(finite)
            res["in_bin_rate"] = sum(lo <= p < hi for p in finite) / len(finite)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generator-checkpoint", required=True)
    ap.add_argument("--predictor-checkpoint", default=None, help="Enables the MAD / in-bin-rate metrics")
    ap.add_argument("--reference-csv", default="data/zinc15_250K.csv", help="Novelty + FCD reference (pretraining pool)")
    ap.add_argument("--train-csv", default=None, help="Dataset train split, for novelty-vs-train")
    ap.add_argument("--bins", default="all", help="'all' or a comma list of bin indices")
    ap.add_argument("--num-samples", type=int, default=10000, help="Total across bins; per-bin = this / n_bins")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--diversity-cap", type=int, default=2000)
    ap.add_argument("--fcd-reference-size", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    gen, vocab, binner, prop_name = _load_generator(args.generator_checkpoint, args.device)
    if binner is None:
        raise ValueError("generator checkpoint has no binner -- it was not fine-tuned conditionally")

    bin_list = list(range(binner.num_bins)) if args.bins == "all" else [int(b) for b in args.bins.split(",")]
    args.per_bin = max(1, args.num_samples // len(bin_list))

    reference = _read_smiles(args.reference_csv) if args.reference_csv else []
    reference_canon = {c for c in (_canon(s) for s in reference) if c} if reference else None
    args.reference_sample = random.sample(reference, min(len(reference), args.fcd_reference_size)) if reference else None
    train_canon = None
    if args.train_csv:
        train_canon = {c for c in (_canon(s) for s in _read_smiles(args.train_csv)) if c}

    predictor = None
    if args.predictor_checkpoint:
        model, mean, scale = _load_predictor(args.predictor_checkpoint, args.device)
        predictor = (model, mean, scale)

    per_bin = [evaluate_bin(gen, vocab, binner, b, args, reference_canon, train_canon, predictor) for b in bin_list]

    def _avg(key):
        xs = [r[key] for r in per_bin if isinstance(r.get(key), (int, float)) and r[key] == r[key]]
        return sum(xs) / len(xs) if xs else None

    summary = {
        "property": prop_name,
        "generator_checkpoint": args.generator_checkpoint,
        "num_bins": binner.num_bins,
        "per_bin_samples": args.per_bin,
        "overall": {k: _avg(k) for k in (
            "validity_selfies", "validity_smiles", "uniqueness", "novelty_vs_reference",
            "novelty_vs_train", "internal_diversity", "unique_scaffolds_frac",
            "scaffold_entropy_norm", "fcd", "mad_vs_bin_center", "in_bin_rate",
        )},
        "per_bin": per_bin,
    }

    print(f"\n=== Generative eval: {prop_name} ===")
    for k, v in summary["overall"].items():
        print(f"  {k:>22s}: {v:.4f}" if isinstance(v, float) else f"  {k:>22s}: {v}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
