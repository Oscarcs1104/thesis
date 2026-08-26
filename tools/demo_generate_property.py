from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data_pipeline.convert_smiles_to_pyg import canonicalize_smiles as _canonical_smiles
from data_pipeline.convert_smiles_to_pyg import smiles_to_data
from data_pipeline.data import load_graph_dataset
from model.model import build_model_from_args
from tools.mol_metrics import mean_pairwise_tanimoto, mols_from_smiles, morgan_fingerprints, murcko_scaffold_smiles, nearest_neighbor_similarity, normalized_shannon_entropy, tanimoto_similarity

# (relative CSV path, target property column)
_DATASET_FILES = {
    "esol": ("data/esol.csv", "logSolubility"),
    "freesolv": ("data/freesolv.csv", "freesolv"),
    "lipo": ("data/lipo.csv", "lipo"),
}


def _lookup_real_property(smiles: str) -> list[tuple[str, str, float]]:
    """Search esol/freesolv/lipo for this exact molecule and return its real label(s)."""
    canonical_input = _canonical_smiles(smiles)
    matches: list[tuple[str, str, float]] = []
    for dataset_name, (rel_path, column) in _DATASET_FILES.items():
        csv_path = ROOT / rel_path
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row_smiles = row.get("smiles")
                if row_smiles is None:
                    continue
                is_match = row_smiles == smiles or (
                    canonical_input is not None and _canonical_smiles(row_smiles) == canonical_input
                )
                if not is_match:
                    continue
                try:
                    matches.append((dataset_name, column, float(row[column])))
                except (TypeError, ValueError):
                    pass
                break
    return matches


def _predict_property(model, smiles: str, device) -> float | None:
    data = smiles_to_data(smiles)
    if data is None:
        return None
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    data = data.to(device)
    with torch.no_grad():
        logits = model(data)
    return float(logits.squeeze(-1).cpu().item())


def build_model(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint.get("args", {})
    args = dict(args)
    args.setdefault("hidden_dim", 256)
    args.setdefault("output_dim", 1)
    args.setdefault("graph_backbone", "gin")
    args.setdefault("language_backbone", "huggingface")
    args.setdefault("language_model_name", "DeepChem/ChemBERTa-77M-MLM")
    args.setdefault("freeze_language_backbone", True)
    args.setdefault("trust_remote_code", False)
    args.setdefault("num_layers", 3)
    args.setdefault("dropout", 0.3)
    args.setdefault("node_encoding", "dense")
    args.setdefault("node_vocab_sizes", [119, 4])
    args.setdefault("use_graph", True)
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
    parser.add_argument(
        "--reference-data-path",
        type=str,
        default=None,
        help="Optional dataset (same format as train.py --data-path) to check novelty/scaffold overlap of the generated candidates against",
    )
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

    real_matches = _lookup_real_property(args.smiles)
    if real_matches:
        for dataset_name, column, value in real_matches:
            print(f"Real value ({dataset_name}, column '{column}'): {value:.4f}")
    else:
        print("Real value: molecule not found in esol/freesolv/lipo")

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

    canonical_input = _canonical_smiles(args.smiles)
    input_mols, _ = mols_from_smiles([canonical_input] if canonical_input else [])
    input_fp = morgan_fingerprints(input_mols)[0] if input_mols else None

    reference_canonical: set[str] | None = None
    if args.reference_data_path:
        reference_graphs = load_graph_dataset(args.reference_data_path)
        reference_canonical = {
            c for c in (_canonical_smiles(getattr(g, "smiles", None) or "") for g in reference_graphs) if c
        }

    seen_candidates: set[str] = set()
    num_invalid = 0

    print("Generated candidates (different molecules targeting a similar property):")
    for idx, candidate in enumerate(candidates, start=1):
        canonical_candidate = _canonical_smiles(candidate)
        if canonical_candidate is None:
            num_invalid += 1
            print(f"{idx:02d}. {candidate}  [INVALID, RDKit could not parse it]")
            continue
        if canonical_candidate == canonical_input:
            print(f"{idx:02d}. {candidate}  [same molecule as the input]")
            continue
        if canonical_candidate in seen_candidates:
            print(f"{idx:02d}. {candidate}  [duplicate of another candidate]")
            continue
        seen_candidates.add(canonical_candidate)
        candidate_pred = _predict_property(model, candidate, device)
        pred_str = f"{candidate_pred:.4f}" if candidate_pred is not None else "N/A"

        tanimoto_str = "N/A"
        if input_fp is not None:
            candidate_mols, _ = mols_from_smiles([canonical_candidate])
            if candidate_mols:
                candidate_fp = morgan_fingerprints(candidate_mols)[0]
                tanimoto_str = f"{tanimoto_similarity(input_fp, candidate_fp):.4f}"

        in_dataset_str = ""
        if reference_canonical is not None:
            in_dataset_str = f"  | already in {args.reference_data_path}: {'yes' if canonical_candidate in reference_canonical else 'no'}"
        print(f"{idx:02d}. {candidate}  | predicted property: {pred_str}  | Tanimoto to input: {tanimoto_str}{in_dataset_str}")

    _print_generation_metrics(list(seen_candidates), len(candidates), num_invalid, canonical_input, args.reference_data_path)


def _print_generation_metrics(canonical_candidates: list[str], num_generated: int, num_invalid: int, canonical_input: str | None, reference_data_path: str | None) -> None:
    print("\n--- Generation metrics ---")
    print(f"Valid: {num_generated - num_invalid}/{num_generated} ({(num_generated - num_invalid) / max(num_generated, 1):.1%})")
    print(f"Unique, novel-vs-input candidates: {len(canonical_candidates)}")

    mols, parse_failed = mols_from_smiles(canonical_candidates)
    if parse_failed:
        print(f"Warning: {parse_failed} canonical candidates failed to re-parse (unexpected)")
    if not mols:
        print("No distinct candidates to measure")
        return

    fps = morgan_fingerprints(mols)

    input_mols, _ = mols_from_smiles([canonical_input] if canonical_input else [])
    if input_mols:
        input_fp = morgan_fingerprints(input_mols)[0]
        input_sims = [tanimoto_similarity(input_fp, fp) for fp in fps]
        print(
            f"Tanimoto similarity to input -- mean: {sum(input_sims) / len(input_sims):.4f}, "
            f"min: {min(input_sims):.4f}, max: {max(input_sims):.4f}"
        )

    if len(mols) < 2:
        print("Not enough distinct candidates for internal diversity/scaffold metrics")
        return

    mean_sim = mean_pairwise_tanimoto(fps)
    print(f"Internal diversity (1 - mean pairwise Tanimoto): {1 - mean_sim:.4f}")

    scaffolds = [murcko_scaffold_smiles(mol) for mol in mols]
    print(f"Unique Murcko scaffolds among candidates: {len(set(scaffolds))}/{len(mols)}")
    normalized_entropy = normalized_shannon_entropy(scaffolds)
    print(f"Normalized Shannon entropy over candidate scaffolds (H/Hmax): {normalized_entropy:.4f} ({normalized_entropy:.1%})")

    if not reference_data_path:
        print("(pass --reference-data-path to also check novelty/scaffold overlap against a training set)")
        return

    reference_graphs = load_graph_dataset(reference_data_path)
    reference_smiles = [s for s in (getattr(g, "smiles", None) for g in reference_graphs) if s]
    reference_mols, _ = mols_from_smiles(reference_smiles)
    reference_fps = morgan_fingerprints(reference_mols)
    reference_scaffolds = {murcko_scaffold_smiles(mol) for mol in reference_mols}

    nn_sims = nearest_neighbor_similarity(fps, reference_fps)
    exact_matches = sum(1 for sim in nn_sims if sim >= 0.999)
    print(f"Reference set: {reference_data_path} ({len(reference_mols)} molecules)")
    print(f"Mean nearest-neighbor Tanimoto similarity to reference: {sum(nn_sims) / len(nn_sims):.4f}")
    print(f"Candidates that exactly match a reference molecule (Tanimoto>=0.999): {exact_matches}/{len(mols)}")
    print(f"Candidates novel (Tanimoto<1.0 to every reference molecule): {len(mols) - exact_matches}/{len(mols)}")

    scaffold_overlap = sum(1 for s in scaffolds if s in reference_scaffolds)
    print(f"Candidate scaffolds also seen in reference set: {scaffold_overlap}/{len(mols)}")


if __name__ == "__main__":
    main()
