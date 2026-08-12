from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.convert_smiles_to_pyg import smiles_to_data
from model.model import build_model_from_args


def build_model(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint.get("args", {})
    args = dict(args)
    args.setdefault("hidden_dim", 256)
    args.setdefault("output_dim", 1)
    args.setdefault("graph_backbone", "gin")
    args.setdefault("language_backbone", "chemberta")
    args.setdefault("language_model_name", "DeepChem/ChemBERTa-77M-MLM")
    args.setdefault("freeze_language_backbone", True)
    args.setdefault("trust_remote_code", False)
    args.setdefault("num_layers", 3)
    args.setdefault("dropout", 0.3)
    args.setdefault("node_encoding", "dense")
    args.setdefault("node_vocab_sizes", [119, 4])
    args.setdefault("use_language", True)
    args.setdefault("use_decoder", True)

    decoder_vocab = checkpoint.get("decoder_vocab")
    if decoder_vocab is None:
        decoder_vocab = {
            "token_to_id": {"<PAD>": 0, "<START>": 1, "<END>": 2, "<OTHER>": 3},
            "id_to_token": {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<OTHER>"},
            "pad_idx": 0,
            "start_idx": 1,
            "end_idx": 2,
            "unk_idx": 3,
        }
        args.setdefault("decoder_vocab_size", 4)
        args.setdefault("decoder_pad_idx", 0)
        args.setdefault("decoder_start_idx", 1)
        args.setdefault("decoder_end_idx", 2)
    else:
        args.setdefault("decoder_vocab_size", len(decoder_vocab["token_to_id"]))
        args.setdefault("decoder_pad_idx", decoder_vocab["pad_idx"])
        args.setdefault("decoder_start_idx", decoder_vocab["start_idx"])
        args.setdefault("decoder_end_idx", decoder_vocab["end_idx"])

    class _Args:
        pass

    loaded_args = _Args()
    loaded_args.__dict__.update(args)

    model = build_model_from_args(loaded_args).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if state_dict is None:
        raise ValueError("Checkpoint does not contain model_state_dict or state_dict")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, decoder_vocab


def main() -> None:
    parser = argparse.ArgumentParser(description="Show a predictor's property estimate and decoder-generated candidates")
    parser.add_argument("--checkpoint-path", required=True, help="Path to the training checkpoint")
    parser.add_argument("--smiles", required=True, help="Input SMILES to convert to a graph")
    parser.add_argument("--property-values", nargs="+", type=float, default=None, help="Optional target property values for generation")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, decoder_vocab = build_model(args.checkpoint_path, str(device))

    data = smiles_to_data(args.smiles)
    if data is None:
        raise ValueError("Invalid SMILES")
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    data = data.to(device)

    with torch.no_grad():
        logits = model(data)
        pred = float(logits.squeeze(-1).cpu().item())

    print(f"Input SMILES: {args.smiles}")
    print(f"Predicted property: {pred:.4f}")

    property_values = None
    if args.property_values is not None:
        property_values = torch.tensor([args.property_values], dtype=torch.float, device=device)
        print(f"Conditioning generation on property: {args.property_values}")
    else:
        property_values = torch.tensor([[pred]], dtype=torch.float, device=device)
        print(f"Conditioning generation on predicted property: {pred:.4f}")

    candidates = model.generate_smiles_candidates(
        data,
        decoder_vocab["id_to_token"],
        max_len=args.max_len,
        num_samples=args.num_samples,
        temperature=args.temperature,
        property_values=property_values,
    )

    print("Generated candidates:")
    for idx, candidate in enumerate(candidates, start=1):
        print(f"{idx:02d}. {candidate}")


if __name__ == "__main__":
    main()
