"""Three diagnostic probes for a HybridJoint checkpoint, side by side per seed molecule
(all with the encoder seeing the seed's real graph+SMILES, unless noted):

  (A) Real property vs. self-predicted property -- condition on the seed's TRUE dataset
      value and, separately, on the model's own regression-head estimate of that same
      seed. Both are close to the seed's OWN value, not a deliberate change.

  (A, +shift) Real property + shift -- condition on true_y + --shift-std-mult train-set
      std deviations (same "-shift/real/+shift" convention as eval_hybrid_control.py) --
      this is the one that actually asks for a DIFFERENT property than the seed has.

  (B) With encoder vs. without encoder -- generate normally (memory = property token +
      graph-node states + SMILES-char states, via build_memory) vs. with the encoder
      bypassed entirely (memory = ONLY the property token, built by hand instead of
      through mola.encode_for_generation). Same decoder weights both times. If removing
      the encoder degrades validity/coherence, the encoder's content -- not just the
      property token -- is doing real work for generation.

Ad-hoc/diagnostic: not part of the benchmark suite. Writes one row per generated sample
to --out (mol_idx, seed_smiles, condition, requested_property, generated_smiles, valid,
tanimoto_to_seed).

Usage:
    python crossmodal_model/generation/probe_encoder.py --dataset freesolv --seed 2025 --mol-indices 0 1
"""
from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd
import torch
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

THIS_DIR = Path(__file__).resolve().parent
TEST_ROOT = THIS_DIR.parent.parent
if str(TEST_ROOT) not in sys.path:
    sys.path.append(str(TEST_ROOT))

from data_pipeline.convert_smiles_to_pyg import smiles_to_data  # noqa: E402
from crossmodal_model.generation.decoder import MoLAConditionalGenerator  # noqa: E402
from crossmodal_model.model.mola_hybrid import HybridMoLA  # noqa: E402
from crossmodal_model.train.core import DATASETS  # noqa: E402
from common.mol_metrics import morgan_fingerprints, tanimoto_similarity  # noqa: E402
from common.repro import TargetStandardizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe: real-vs-predicted conditioning, and with-vs-without encoder")
    parser.add_argument("--dataset", type=str, default="freesolv", choices=list(DATASETS.keys()))
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--mol-indices", type=int, nargs="*", default=[0, 1], help="Indices into the test split to probe")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--shift-std-mult", type=float, default=1.0, help="Positive shift requested = seed's real property + this many train-set std deviations")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    if args.checkpoint_path is None:
        args.checkpoint_path = str(TEST_ROOT / "checkpoints" / "crossmodal" / "hybrid_joint" / f"{args.dataset}_mola_hybrid_joint_s{args.seed}.pt")
    if args.out is None:
        args.out = str(TEST_ROOT / "results" / "mola" / f"probe_encoder_{args.dataset}.csv")
    return args


def featurize_single(smiles: str, char_vocab: dict, max_sm_len: int, device):
    d = smiles_to_data(smiles)
    if d is None or d.edge_index.size(1) == 0:
        return None
    sm_idx = [char_vocab.get(ch, 0) for ch in smiles[:max_sm_len]]
    if len(sm_idx) < max_sm_len:
        sm_idx.extend([0] * (max_sm_len - len(sm_idx)))
    d.sm = torch.tensor(sm_idx, dtype=torch.long).unsqueeze(0)
    d.batch = torch.zeros(d.x.size(0), dtype=torch.long)
    return d.to(device)


def generate_batch(generator, seed_data, vocab, prop, n, max_len, temperature, device):
    out = []
    for _ in range(n):
        smi = generator.generate(seed_data, vocab, property_values=prop, max_len=max_len, temperature=temperature, sample=True)
        out.append(smi)
    return out


def generate_no_encoder(generator, vocab, prop, n, max_len, temperature, device):
    """Same decoder, memory = only the property token (property_proj(prop)), no
    graph/SMILES states at all -- built by hand instead of via build_memory."""
    prop_token = generator.property_proj(prop.to(generator.property_proj[0].weight.dtype)).unsqueeze(1)  # [1,1,H]
    pad_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)
    out = []
    for _ in range(n):
        smi = generator.decoder.generate(prop_token, pad_mask, vocab, max_len=max_len, temperature=temperature, sample=True)
        out.append(smi)
    return out


def rows_for(mol_idx: int, seed_smiles: str, condition: str, requested_property: float, smiles_list, seed_fp) -> list:
    rows = []
    for raw_smi in smiles_list:
        mol = Chem.MolFromSmiles(raw_smi) if raw_smi else None
        valid = mol is not None
        canonical = Chem.MolToSmiles(mol) if valid else ""
        tanimoto = None
        if valid:
            fp = morgan_fingerprints([mol])[0]
            tanimoto = tanimoto_similarity(seed_fp, fp)
        rows.append({
            "mol_idx": mol_idx, "seed_smiles": seed_smiles, "condition": condition,
            "requested_property": requested_property, "generated_smiles": canonical,
            "valid": valid, "tanimoto_to_seed": tanimoto,
        })
    return rows


def summarize(label: str, rows: list) -> None:
    valid_rows = [r for r in rows if r["valid"]]
    unique = {r["generated_smiles"] for r in valid_rows}
    print(f"  {label}: {len(valid_rows)}/{len(rows)} validas, {len(unique)} unicas")
    for r in valid_rows:
        print(f"    {r['generated_smiles']}  (Tanimoto al seed={r['tanimoto_to_seed']:.2f})")
    if len(valid_rows) < len(rows):
        print(f"    [{len(rows) - len(valid_rows)} invalidas omitidas]")


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
    print(f"Cargado {args.checkpoint_path}\n")

    cfg = DATASETS[args.dataset]
    csv_dir = TEST_ROOT / "data" / "deepchem_molnet" / cfg["dir"] / "csv"
    train_y = pd.read_csv(csv_dir / "train.csv")[cfg["target_col"]].astype(float).to_numpy()
    test_df = pd.read_csv(csv_dir / "test.csv")
    standardizer = TargetStandardizer(enabled=True).fit(torch.tensor(train_y, dtype=torch.float32).view(-1, 1))
    max_len = saved_args["max_selfies_len"]
    train_std = float(train_y.std())
    shift = args.shift_std_mult * train_std
    print(f"Train property std={train_std:.3f} -> shift positivo = +{shift:.3f} ({args.shift_std_mult} std)\n")
    all_rows: list = []

    for mi in args.mol_indices:
        seed_smi = str(test_df["smiles"].iloc[mi])
        true_y = float(test_df[cfg["target_col"]].iloc[mi])
        seed_mol = Chem.MolFromSmiles(seed_smi)
        if seed_mol is None:
            print(f"[{mi}] SMILES invalido, salto")
            continue
        seed_canonical = Chem.MolToSmiles(seed_mol)
        seed_fp = morgan_fingerprints([seed_mol])[0]
        seed_data = featurize_single(seed_canonical, char_vocab, saved_args["max_sm_len"], device)
        if seed_data is None:
            print(f"[{mi}] no featurizable (sin enlaces), salto")
            continue

        with torch.no_grad():
            pred_y = float(standardizer.inverse_transform(mola(seed_data)[-1]).view(-1)[0].item())

        print("=" * 78)
        print(f"[{mi}] Semilla: {seed_canonical}")
        print(f"  Propiedad real (dataset)     = {true_y:.3f}")
        print(f"  Propiedad autopredicha (mola) = {pred_y:.3f}  (diferencia = {pred_y - true_y:+.3f})")
        print()

        print(f"(A) Condicionando en la propiedad REAL ({true_y:.3f}), CON encoder:")
        prop_real = torch.tensor([[true_y]], dtype=torch.float, device=device)
        rows_real = rows_for(mi, seed_canonical, "real_con_encoder", true_y, generate_batch(generator, seed_data, selfies_vocab, prop_real, args.n_samples, max_len, args.temperature, device), seed_fp)
        summarize("real", rows_real)
        all_rows.extend(rows_real)

        print(f"\n(A) Condicionando en la propiedad AUTOPREDICHA ({pred_y:.3f}), CON encoder:")
        prop_pred = torch.tensor([[pred_y]], dtype=torch.float, device=device)
        rows_pred = rows_for(mi, seed_canonical, "predicha_con_encoder", pred_y, generate_batch(generator, seed_data, selfies_vocab, prop_pred, args.n_samples, max_len, args.temperature, device), seed_fp)
        summarize("predicha", rows_pred)
        all_rows.extend(rows_pred)

        shifted_y = true_y + shift
        print(f"\n(A) Condicionando en la propiedad REAL + shift positivo ({shifted_y:.3f} = {true_y:.3f} + {shift:.3f}), CON encoder:")
        prop_shift = torch.tensor([[shifted_y]], dtype=torch.float, device=device)
        rows_shift = rows_for(mi, seed_canonical, "real_mas_shift_con_encoder", shifted_y, generate_batch(generator, seed_data, selfies_vocab, prop_shift, args.n_samples, max_len, args.temperature, device), seed_fp)
        summarize("real+shift", rows_shift)
        all_rows.extend(rows_shift)

        print(f"\n(B) Condicionando en la propiedad REAL ({true_y:.3f}), SIN encoder (memoria = solo el token de propiedad):")
        rows_no_enc = rows_for(mi, seed_canonical, "real_sin_encoder", true_y, generate_no_encoder(generator, selfies_vocab, prop_real, args.n_samples, max_len, args.temperature, device), seed_fp)
        summarize("sin encoder", rows_no_enc)
        all_rows.extend(rows_no_enc)
        print()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mol_idx", "seed_smiles", "condition", "requested_property", "generated_smiles", "valid", "tanimoto_to_seed"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Guardado {len(all_rows)} filas en {out_path}")


if __name__ == "__main__":
    main()
