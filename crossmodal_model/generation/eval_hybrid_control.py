"""Structural-control + property-control evaluation for a HybridMoLA generator
(train_hybrid.py's checkpoint): does asking for a different property produce a
structurally different molecule with the requested property, or just a perturbed
reconstruction? Same metrics as the MoLA-original check: Tanimoto to seed, directional
control (+shift vs -shift), requested-vs-estimated correlation, intra-sample diversity.

Property estimation uses the checkpoint's OWN regression head (self-scoring, not an
independent predictor) -- treat these numbers as self-consistency, not a final result.

--prefix-len N (0 = default/free mode): forces the decoder to COPY the seed's own first N
SELFIES tokens before sampling freely, instead of generating from scratch. Trades property
control for structural fidelity -- measured on FreeSolv (test set):
    N=0 (free):  Tanimoto-to-seed=0.224  directional control=93.8%  Pearson=0.684
    N=4:         Tanimoto-to-seed=0.315  directional control=87.5%  Pearson=0.618
    N=8:         Tanimoto-to-seed=0.495  directional control=68.8%  Pearson=0.546
N=4 is a reasonable middle ground; there's no free lunch (higher N always costs some
directional control) -- pick based on whether the application needs "similar analog with
roughly the right property" or "confidently the right property, less constrained shape".

Usage:
    python crossmodal_model/generation/eval_hybrid_control.py --dataset freesolv --seed 2025
    python crossmodal_model/generation/eval_hybrid_control.py --dataset freesolv --seed 2025 --prefix-len 4
"""
from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path
from typing import Dict, List

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

THIS_DIR = Path(__file__).resolve().parent
TEST_ROOT = THIS_DIR.parent.parent
if str(TEST_ROOT) not in sys.path:
    sys.path.append(str(TEST_ROOT))

from scipy.stats import spearmanr  # noqa: E402

from data_pipeline.convert_smiles_to_pyg import smiles_to_data  # noqa: E402
from crossmodal_model.generation.decoder import MoLAConditionalGenerator  # noqa: E402
from crossmodal_model.model.mola_hybrid import HybridMoLA  # noqa: E402
from crossmodal_model.train.core import DATASETS  # noqa: E402
from common.mol_metrics import mean_pairwise_tanimoto, morgan_fingerprints, tanimoto_similarity  # noqa: E402
from common.repro import TargetStandardizer  # noqa: E402
from common.selfies_vocab import decode_ids, tokenize_molecule  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Structural/property control check for a HybridMoLA generator")
    parser.add_argument("--dataset", type=str, default="freesolv", choices=list(DATASETS.keys()))
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--shift-std-mult", type=float, default=1.0)
    parser.add_argument("--n-samples", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--prefix-len", type=int, default=0, help="Force the decoder to copy this many of the seed's own SELFIES tokens before sampling freely -- 0 disables (normal generation)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    if args.checkpoint_path is None:
        args.checkpoint_path = str(TEST_ROOT / "checkpoints" / "crossmodal" / "hybrid_joint" / f"{args.dataset}_mola_hybrid_joint_s{args.seed}.pt")
    if args.out is None:
        args.out = str(TEST_ROOT / "results" / "mola" / f"mola_hybrid_{args.dataset}_control_eval.csv")
    return args


def featurize_single(smiles: str, char_vocab: dict, max_sm_len: int, device):
    d = smiles_to_data(smiles)
    if d is None:
        return None
    if d.edge_index.size(1) == 0:
        # Single heavy atom, no bonds -- GINEConv's edge_dim projection needs a real
        # edge_attr tensor even for empty graphs; thesis_model's EdgeFeatureEncoder
        # returns None for numel()==0, which only doesn't crash when this molecule is
        # batched alongside others that DO have edges (training/eval loops always are).
        # Evaluated alone (this script's one-seed-at-a-time loop), it crashes -- skip it.
        return None
    sm_idx = [char_vocab.get(ch, 0) for ch in smiles[:max_sm_len]]
    if len(sm_idx) < max_sm_len:
        sm_idx.extend([0] * (max_sm_len - len(sm_idx)))
    d.sm = torch.tensor(sm_idx, dtype=torch.long).unsqueeze(0)
    d.batch = torch.zeros(d.x.size(0), dtype=torch.long)
    return d.to(device)


@torch.no_grad()
def generate_with_prefix(generator: MoLAConditionalGenerator, seed_data, vocab: dict, prop: torch.Tensor, seed_smiles: str, prefix_len: int, max_len: int, temperature: float) -> str:
    """Same decoder/memory as generator.generate, but the first `prefix_len` SELFIES
    tokens are forced to be the seed's OWN tokens (copied, not sampled) -- guarantees a
    literal shared prefix instead of hoping similarity emerges from cross-attention alone."""
    memory, memory_pad_mask = generator._memory(seed_data, prop)
    seed_tokens = tokenize_molecule(seed_smiles)
    prefix_ids = [vocab["token_to_id"].get(t, vocab["unk_idx"]) for t in seed_tokens[:prefix_len]]
    generated = [vocab["start_idx"]] + prefix_ids
    for _ in range(max(max_len - len(generated), 0)):
        ids = torch.tensor([generated], dtype=torch.long, device=memory.device)
        logits = generator.decoder.forward(memory, memory_pad_mask, ids)[:, -1] / max(temperature, 1e-6)
        next_id = int(torch.multinomial(torch.softmax(logits, dim=-1), 1).item())
        if next_id == vocab["end_idx"]:
            break
        generated.append(next_id)
    return decode_ids(generated[1:], vocab["id_to_token"])


def self_predict(mola: HybridMoLA, standardizer: TargetStandardizer, data) -> float:
    with torch.no_grad():
        out = mola(data)[-1]
        raw = standardizer.inverse_transform(out)
    return float(raw.view(-1)[0].item())


def main() -> None:
    args = parse_args()
    device = args.device

    ckpt = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    char_vocab, selfies_vocab, saved_args = ckpt["char_vocab"], ckpt["selfies_vocab"], ckpt["args"]

    mola = HybridMoLA(
        sm_vocab_size=len(char_vocab), hidden_dim=saved_args["hidden_dim"], output_dim=1,
        num_layers=saved_args["num_layers"], positional_smiles=True, max_sm_len=saved_args["max_sm_len"],
    )
    generator = MoLAConditionalGenerator(
        mola, vocab_size=len(selfies_vocab["token_to_id"]), hidden_dim=saved_args["hidden_dim"],
        pad_idx=selfies_vocab["pad_idx"], use_property=True, decoder_layers=saved_args["decoder_layers"],
        max_len=saved_args["max_selfies_len"],
    ).to(device)
    generator.load_state_dict(ckpt["model_state_dict"])
    generator.eval()
    print(f"Loaded {args.checkpoint_path}")

    cfg = DATASETS[args.dataset]
    csv_dir = TEST_ROOT / "data" / "deepchem_molnet" / cfg["dir"] / "csv"
    train_df = pd.read_csv(csv_dir / "train.csv")
    test_df = pd.read_csv(csv_dir / "test.csv")
    train_smiles = train_df["smiles"].astype(str).tolist()
    train_y = train_df[cfg["target_col"]].astype(float).to_numpy()
    test_smiles = test_df["smiles"].astype(str).tolist()
    test_y = test_df[cfg["target_col"]].astype(float).to_numpy()

    train_std = float(train_y.std())
    standardizer = TargetStandardizer(enabled=True).fit(torch.tensor(train_y, dtype=torch.float32).view(-1, 1))
    train_canonical = set()
    for smi in train_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            train_canonical.add(Chem.MolToSmiles(mol))

    shift = args.shift_std_mult * train_std
    conditions = {"-shift": -shift, "real": 0.0, "+shift": shift}
    print(f"Train property std={train_std:.3f} -> shift=+/-{shift:.3f} ({args.shift_std_mult} std)")

    rows: List[Dict] = []
    print(f"Generating {args.n_samples} samples x {len(conditions)} conditions x {len(test_smiles)} test molecules...")
    for mi, seed_smi in enumerate(test_smiles):
        seed_mol = Chem.MolFromSmiles(seed_smi)
        if seed_mol is None:
            continue
        seed_fp = morgan_fingerprints([seed_mol])[0]
        true_y = float(test_y[mi])

        seed_data = featurize_single(seed_smi, char_vocab, saved_args["max_sm_len"], device)
        if seed_data is None:
            continue

        for cond_name, cond_shift in conditions.items():
            requested = true_y + cond_shift
            prop = torch.tensor([[requested]], dtype=torch.float, device=device)
            for k in range(args.n_samples):
                if args.prefix_len > 0:
                    generated = generate_with_prefix(generator, seed_data, selfies_vocab, prop, seed_smi, args.prefix_len, saved_args["max_selfies_len"], args.temperature)
                else:
                    generated = generator.generate(seed_data, selfies_vocab, property_values=prop, max_len=saved_args["max_selfies_len"], temperature=args.temperature, sample=True)
                mol = Chem.MolFromSmiles(generated) if generated else None
                canonical = Chem.MolToSmiles(mol) if mol is not None else ""
                tanimoto_to_seed = predicted = None
                if mol is not None:
                    fp = morgan_fingerprints([mol])[0]
                    tanimoto_to_seed = tanimoto_similarity(seed_fp, fp)
                    pred_data = featurize_single(canonical, char_vocab, saved_args["max_sm_len"], device)
                    if pred_data is not None:
                        predicted = self_predict(mola, standardizer, pred_data)
                rows.append({
                    "mol_idx": mi, "seed_smiles": seed_smi, "condition": cond_name,
                    "requested_property": requested, "generated_canonical": canonical,
                    "valid": mol is not None, "tanimoto_to_seed": tanimoto_to_seed,
                    "self_predicted_property": predicted,
                    "novel": (canonical not in train_canonical) if mol is not None else None,
                })
        if (mi + 1) % 16 == 0:
            print(f"  ...{mi + 1}/{len(test_smiles)} molecules done")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {out_path}")

    valid_rows = [r for r in rows if r["valid"]]
    scored_rows = [r for r in valid_rows if r["self_predicted_property"] is not None]
    print(f"\nValidity: {len(valid_rows)}/{len(rows)} ({len(valid_rows)/len(rows):.1%})")

    tan = np.array([r["tanimoto_to_seed"] for r in valid_rows if r["tanimoto_to_seed"] is not None])
    print(f"Tanimoto to seed (1.0=identical): mean={tan.mean():.3f} std={tan.std():.3f}")
    print(f"  -> fraction identical to seed: {(tan >= 0.999).mean():.1%}")

    requested = np.array([r["requested_property"] for r in scored_rows])
    predicted = np.array([r["self_predicted_property"] for r in scored_rows])
    pearson_r = float(np.corrcoef(requested, predicted)[0, 1]) if len(scored_rows) > 2 else float("nan")
    spearman_r = float(spearmanr(requested, predicted).correlation) if len(scored_rows) > 2 else float("nan")
    print(f"\nRequested vs. self-predicted property: Pearson r={pearson_r:.3f} Spearman r={spearman_r:.3f}")
    print("  (self-consistency only -- not an independent check)")

    print(f"\nPer-condition:")
    print(f"  {'condition':>8s} {'n_valid':>8s} {'mean_prop':>10s} {'mean_tanimoto':>14s} {'novelty':>8s}")
    for cond_name in conditions:
        cond_scored = [r for r in scored_rows if r["condition"] == cond_name]
        cond_valid = [r for r in valid_rows if r["condition"] == cond_name]
        if cond_scored:
            props = np.array([r["self_predicted_property"] for r in cond_scored])
            tans = np.array([r["tanimoto_to_seed"] for r in cond_valid if r["tanimoto_to_seed"] is not None])
            novel_frac = np.mean([bool(r["novel"]) for r in cond_valid]) if cond_valid else float("nan")
            print(f"  {cond_name:>8s} {len(cond_scored):8d} {props.mean():10.3f} {tans.mean():14.3f} {novel_frac:7.1%}")

    print(f"\nDirectional control (+shift vs -shift, per molecule):")
    n_both = n_correct = 0
    for mi in range(len(test_smiles)):
        minus = np.array([r["self_predicted_property"] for r in scored_rows if r["mol_idx"] == mi and r["condition"] == "-shift"])
        plus = np.array([r["self_predicted_property"] for r in scored_rows if r["mol_idx"] == mi and r["condition"] == "+shift"])
        if len(minus) == 0 or len(plus) == 0:
            continue
        n_both += 1
        if plus.mean() > minus.mean():
            n_correct += 1
    print(f"  {n_correct}/{n_both} molecules ({n_correct/max(n_both,1):.1%}) have mean(+shift) > mean(-shift)")

    print(f"\nStructural diversity among the {args.n_samples} candidates per (molecule, condition):")
    for cond_name in conditions:
        diversities = []
        for mi in range(len(test_smiles)):
            fps = [
                morgan_fingerprints([Chem.MolFromSmiles(r["generated_canonical"])])[0]
                for r in valid_rows if r["mol_idx"] == mi and r["condition"] == cond_name and r["generated_canonical"]
            ]
            div = mean_pairwise_tanimoto(fps)
            if div is not None:
                diversities.append(div)
        if diversities:
            arr = np.array(diversities)
            print(f"  {cond_name:>8s}: mean intra-sample Tanimoto={arr.mean():.3f}")

    print("\n=== CONCLUSION ===")
    print(f"Tanimoto to seed (mean): {tan.mean():.3f}")
    print(f"Directional control: {n_correct/max(n_both,1):.1%} of molecules move the right way")


if __name__ == "__main__":
    main()
