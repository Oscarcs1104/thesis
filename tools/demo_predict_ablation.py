"""Predict a property with a checkpoint from one of the predictor-only fusion
ablations (training/train_cross_attention.py or training/train_moe_fusion.py).
Auto-detects which architecture the checkpoint is from via its saved args.

See COMMANDS.md for usage.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data_pipeline.convert_smiles_to_pyg import smiles_to_data
from thesis_model.generation.demo_generate_property import _lookup_real_property


def build_ablation_model(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args_dict = dict(checkpoint.get("args", {}))
    args_dict.setdefault("output_dim", 1)
    args_dict.setdefault("hidden_dim", 256)
    args_dict.setdefault("graph_backbone", "gin")
    args_dict.setdefault("num_layers", 3)
    args_dict.setdefault("dropout", 0.3)
    args_dict.setdefault("node_encoding", "dense")
    args_dict.setdefault("node_vocab_sizes", [119, 4])
    args_dict.setdefault("language_model_name", "DeepChem/ChemBERTa-77M-MLM")
    args_dict.setdefault("freeze_language_backbone", True)
    args_dict.setdefault("trust_remote_code", False)

    class _Args:
        pass

    args = _Args()
    args.__dict__.update(args_dict)

    if "num_heads" in args_dict:
        from thesis_model.model.cross_attention_model import build_cross_attention_model_from_args

        model_type = "cross_attention"
        args_dict.setdefault("num_cross_layers", 1)
        args.__dict__.update(args_dict)
        model = build_cross_attention_model_from_args(args)
    elif "num_experts" in args_dict:
        from thesis_model.model.moe_fusion_model import build_moe_model_from_args

        model_type = "moe"
        model = build_moe_model_from_args(args)
    else:
        raise ValueError(f"Cannot determine ablation model type from checkpoint args: {sorted(args_dict.keys())}")

    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device).eval()
    return model, model_type


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a property with a fusion-ablation checkpoint (cross-attention or MoE)")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, model_type = build_ablation_model(args.checkpoint_path, str(device))
    print(f"Loaded {model_type} model from {args.checkpoint_path}")

    data = smiles_to_data(args.smiles)
    if data is None:
        raise ValueError("Invalid SMILES")
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    data = data.to(device)

    with torch.no_grad():
        pred = float(model(data).squeeze(-1).cpu().item())

    print(f"Input SMILES: {args.smiles}")
    print(f"Predicted property: {pred:.4f}")

    real_matches = _lookup_real_property(args.smiles)
    if real_matches:
        for dataset_name, column, value in real_matches:
            print(f"Real value ({dataset_name}, column '{column}'): {value:.4f}")
    else:
        print("Real value: molecule not found in esol/freesolv/lipo")


if __name__ == "__main__":
    main()
