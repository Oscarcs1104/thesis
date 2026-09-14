"""Ad-hoc demo for HybridJoint: given ONLY a seed SMILES (no ground-truth property
needed -- the seed doesn't even have to be in the dataset), the model first PREDICTS its
own property with its regression head, conditions generation on that self-predicted
value (optionally shifted +/-), and reports the self-predicted property of each
generated candidate. Fully self-consistent: no independent predictor, no dataset lookup
for the seed's real value.

Usage:
    python crossmodal_model/generation/from_smiles_hybrid.py --smiles "CCC#N" --dataset freesolv --n-samples 20
    python crossmodal_model/generation/from_smiles_hybrid.py --smiles "CCC#N" --dataset freesolv --shift-std-mult 1.0
"""
from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Self-consistent demo: predict seed property, condition on it, generate")
    parser.add_argument("--smiles", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="freesolv", choices=list(DATASETS.keys()), help="Which HybridJoint checkpoint/train stats to use")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--shift-std-mult", type=float, default=0.0, help="0 = only condition on the self-predicted value; >0 also tries +-shift*std around it")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.checkpoint_path is None:
        args.checkpoint_path = str(TEST_ROOT / "checkpoints" / "crossmodal" / "hybrid_joint" / f"{args.dataset}_mola_hybrid_joint_s{args.seed}.pt")
    return args


def featurize(smiles: str, char_vocab: dict, max_sm_len: int, device):
    d = smiles_to_data(smiles)
    if d is None or d.edge_index.size(1) == 0:
        return None
    sm_idx = [char_vocab.get(ch, 0) for ch in smiles[:max_sm_len]]
    if len(sm_idx) < max_sm_len:
        sm_idx.extend([0] * (max_sm_len - len(sm_idx)))
    d.sm = torch.tensor(sm_idx, dtype=torch.long).unsqueeze(0)
    d.batch = torch.zeros(d.x.size(0), dtype=torch.long)
    return d.to(device)


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

    # Standardizer: rebuilt from the same official train split this checkpoint was
    # trained on (deterministic -- same recipe eval_hybrid_control.py uses).
    cfg = DATASETS[args.dataset]
    train_y = pd.read_csv(TEST_ROOT / "data" / "deepchem_molnet" / cfg["dir"] / "csv" / "train.csv")[cfg["target_col"]].astype(float).to_numpy()
    standardizer = TargetStandardizer(enabled=True).fit(torch.tensor(train_y, dtype=torch.float32).view(-1, 1))
    train_std = float(train_y.std())

    seed_mol = Chem.MolFromSmiles(args.smiles)
    if seed_mol is None:
        raise ValueError(f"'{args.smiles}' is not a valid SMILES")
    canonical_seed = Chem.MolToSmiles(seed_mol)
    seed_fp = morgan_fingerprints([seed_mol])[0]
    seed_data = featurize(canonical_seed, char_vocab, saved_args["max_sm_len"], device)
    if seed_data is None:
        raise ValueError(f"Could not featurize '{canonical_seed}' (e.g. no bonds)")

    # 1) The model predicts the SEED's own property -- no ground truth used anywhere.
    with torch.no_grad():
        seed_pred = float(standardizer.inverse_transform(mola(seed_data)[-1]).view(-1)[0].item())
    print(f"Semilla: {args.smiles!r} -> canonica {canonical_seed!r}")
    print(f"Propiedad AUTOPREDICHA de la semilla (no es el valor real del dataset): {seed_pred:.3f}\n")

    shift = args.shift_std_mult * train_std
    conditions = {"real (=autopredicha)": 0.0}
    if shift > 0:
        conditions = {"-shift": -shift, **conditions, "+shift": shift}

    for cond_name, delta in conditions.items():
        target = seed_pred + delta
        prop = torch.tensor([[target]], dtype=torch.float, device=device)
        print(f"=== Condicion {cond_name}: pido propiedad = {target:.3f} ===")
        seen = set()
        for k in range(args.n_samples):
            generated = generator.generate(seed_data, selfies_vocab, property_values=prop, max_len=saved_args["max_selfies_len"], temperature=args.temperature, sample=True)
            mol = Chem.MolFromSmiles(generated) if generated else None
            if mol is None:
                print(f"  [{k:2d}] INVALIDO")
                continue
            canon = Chem.MolToSmiles(mol)
            cand_data = featurize(canon, char_vocab, saved_args["max_sm_len"], device)
            predicted = None
            if cand_data is not None:
                with torch.no_grad():
                    predicted = float(standardizer.inverse_transform(mola(cand_data)[-1]).view(-1)[0].item())
            fp = morgan_fingerprints([mol])[0]
            tan = tanimoto_similarity(seed_fp, fp)
            dup = " (repetida)" if canon in seen else ""
            seen.add(canon)
            pred_str = f"propiedad autopredicha={predicted:7.3f}" if predicted is not None else "propiedad autopredicha=  N/A"
            print(f"  [{k:2d}] {canon:30s} {pred_str} | Tanimoto al seed={tan:.2f}{dup}")
        print()


if __name__ == "__main__":
    main()
