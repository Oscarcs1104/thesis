"""Ad-hoc generation test: dame un SMILES semilla + uno o mas valores de propiedad
objetivo, y te devuelve candidatos generados (muestreo, no solo greedy), validados con
RDKit, y -- si el predictor independiente esta disponible -- su propiedad predicha.

Reusa el checkpoint ya entrenado (freesolv_mola_gen_s2025.pt por defecto) y el mismo
predictor independiente del chequeo semantico (eval_conditioning.py).

Uso:
    python crossmodal_model/generation/from_smiles.py --smiles "CC(C)C(C)C" --property-values -8 -3.8 2
    python crossmodal_model/generation/from_smiles.py --smiles "c1ccccc1O" --property-values -5 --num-samples 10 --temperature 1.2
    python crossmodal_model/generation/from_smiles.py --smiles "CCO" --property-values -3.8 --no-predict   # sin el predictor independiente
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

THIS_DIR = Path(__file__).resolve().parent
TEST_ROOT = THIS_DIR.parent.parent
if str(TEST_ROOT) not in sys.path:
    sys.path.append(str(TEST_ROOT))

import deepchem as dc  # noqa: E402
from torch_geometric.data import Data  # noqa: E402

from crossmodal_model.data.featurize import build_vocab as build_char_vocab  # noqa: E402
from crossmodal_model.generation.decoder import MoLAConditionalGenerator  # noqa: E402
from crossmodal_model.model.mola import MoLA  # noqa: E402
from crossmodal_model.generation.eval_conditioning import (  # noqa: E402 -- reuse the exact same independent predictor
    DATASET, PREDICTOR_MAX_SM_LEN,
    load_independent_predictor, predict_property,
)
from crossmodal_model.train.core import DATASETS, load_fixed_split  # noqa: E402

GEN_CKPT_PATH_DEFAULT = TEST_ROOT / "checkpoints" / "crossmodal" / "generation" / "freesolv_mola_gen_s2025.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate molecules from a seed SMILES + target property value(s)")
    parser.add_argument("--smiles", type=str, required=True, help="Seed molecule (any valid SMILES)")
    parser.add_argument("--property-values", type=float, nargs="+", required=True, help="One or more target property values (raw units, e.g. hydration free energy for FreeSolv)")
    parser.add_argument("--num-samples", type=int, default=8, help="Sampled candidates per property value")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--checkpoint-path", type=str, default=str(GEN_CKPT_PATH_DEFAULT))
    parser.add_argument("--no-predict", dest="predict", action="store_false", help="Skip scoring candidates with the independent predictor")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_seed_data(smiles: str, featurizer, char_vocab: dict, max_sm_len: int, device):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"'{smiles}' is not a valid SMILES")
    canonical = Chem.MolToSmiles(mol)

    graph = featurizer.featurize([canonical])[0]
    if not (hasattr(graph, "node_features") and hasattr(graph, "edge_index")):
        raise ValueError(f"MolGraphConvFeaturizer could not featurize '{canonical}' (e.g. single heavy atom -- no bonds)")

    x = torch.tensor(graph.node_features, dtype=torch.float32)
    edge_index = torch.tensor(graph.edge_index, dtype=torch.long)
    sm_idx = [char_vocab.get(ch, 0) for ch in canonical[:max_sm_len]]
    if len(sm_idx) < max_sm_len:
        sm_idx.extend([0] * (max_sm_len - len(sm_idx)))
    sm = torch.tensor(sm_idx, dtype=torch.long).unsqueeze(0)
    data = Data(x=x, edge_index=edge_index, sm=sm)
    data.batch = torch.zeros(x.size(0), dtype=torch.long)
    return data.to(device), canonical


def main() -> None:
    args = parse_args()
    device = args.device

    ckpt = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    char_vocab, selfies_vocab, saved_args = ckpt["char_vocab"], ckpt["selfies_vocab"], ckpt["args"]

    mola = MoLA(
        graph_dim=30, sm_vocab_size=len(char_vocab), hidden_dim=saved_args["hidden_dim"],
        output_dim=1, num_layers=saved_args["num_layers"], positional_smiles=True,
        max_sm_len=saved_args["max_sm_len"],
    )
    generator = MoLAConditionalGenerator(
        mola, vocab_size=len(selfies_vocab["token_to_id"]), hidden_dim=saved_args["hidden_dim"],
        pad_idx=selfies_vocab["pad_idx"], use_property=True, decoder_layers=saved_args["decoder_layers"],
        max_len=saved_args["max_selfies_len"],
    ).to(device)
    generator.load_state_dict(ckpt["model_state_dict"])
    generator.eval()

    featurizer = dc.feat.MolGraphConvFeaturizer()
    seed_data, canonical_smiles = build_seed_data(args.smiles, featurizer, char_vocab, saved_args["max_sm_len"], device)
    print(f"Seed molecule: {args.smiles!r} -> canonical {canonical_smiles!r}")

    predictor = standardizer = predictor_char_vocab = None
    if args.predict:
        cfg = DATASETS[DATASET]
        csv_dir = TEST_ROOT / "data" / "deepchem_molnet" / cfg["dir"] / "csv"
        train_ds = load_fixed_split(csv_dir / "train.csv", cfg["target_col"], dc.feat.MolGraphConvFeaturizer())
        valid_ds = load_fixed_split(csv_dir / "valid.csv", cfg["target_col"], dc.feat.MolGraphConvFeaturizer())
        test_ds = load_fixed_split(csv_dir / "test.csv", cfg["target_col"], dc.feat.MolGraphConvFeaturizer())
        train_y_raw = np.asarray(train_ds.y, dtype="float64").reshape(-1)
        predictor_char_vocab = build_char_vocab(list(train_ds.ids) + list(valid_ds.ids) + list(test_ds.ids))
        predictor, standardizer = load_independent_predictor(train_y_raw, predictor_char_vocab)
        print(f"Independent predictor loaded ({DATASET} train mean={train_y_raw.mean():.3f} std={train_y_raw.std():.3f})\n")

    for target_value in args.property_values:
        print(f"=== target property = {target_value:.3f} ===")
        prop = torch.tensor([[target_value]], dtype=torch.float, device=device)
        seen = set()
        for k in range(args.num_samples):
            generated = generator.generate(seed_data, selfies_vocab, property_values=prop, max_len=saved_args["max_selfies_len"], temperature=args.temperature, sample=True)
            mol = Chem.MolFromSmiles(generated) if generated else None
            if mol is None:
                print(f"  [{k}] INVALID  raw={generated!r}")
                continue
            canonical = Chem.MolToSmiles(mol)
            dup = " (duplicate)" if canonical in seen else ""
            seen.add(canonical)
            pred_str = ""
            if args.predict:
                pred_data = build_seed_data(canonical, dc.feat.MolGraphConvFeaturizer(), predictor_char_vocab, PREDICTOR_MAX_SM_LEN, device)[0]
                predicted = predict_property(predictor, standardizer, pred_data)
                pred_str = f"  predicted={predicted:8.3f}"
            print(f"  [{k}] {canonical:40s}{pred_str}{dup}")
        print()


if __name__ == "__main__":
    main()
