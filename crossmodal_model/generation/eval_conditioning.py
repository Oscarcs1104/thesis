"""Semantic conditioning check (no retraining, no architecture change): does the
requested property value actually steer the PREDICTED property of the generated
molecules (measured by an INDEPENDENT model), or does the generator only produce
different-but-property-blind molecules when the token changes?

Independent predictor: checkpoints/crossmodal/mola_graph_smiles_a2_smiles_fix/
freesolv_..._s2025.pt -- a SEPARATELY trained MoLA G+S regression model
(positional_smiles=True, same featurization recipe as the generator's shared encoder,
but a different weight instance that never saw the generation task). Not the generative
model itself, per the request ("idealmente no uses el propio modelo generativo").

For every FreeSolv test molecule: generate N candidates conditioned on (true_y - 2*std),
true_y, and (true_y + 2*std) -- same sampling strategy/temperature/N for all three.
Filter to RDKit-valid candidates, predict their property with the independent model,
and report requested-vs-predicted correlation, per-condition distributions, the
"+2std > -2std" directional check, and validity/uniqueness/novelty per condition.
"""
from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path
from typing import Dict, List

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

from crossmodal_model.data.featurize import build_vocab as build_char_vocab, prepare_data  # noqa: E402
from crossmodal_model.generation.decoder import MoLAConditionalGenerator  # noqa: E402
from crossmodal_model.model.mola import MoLA  # noqa: E402
from crossmodal_model.train.core import DATASETS, load_fixed_split  # noqa: E402
from common.mol_metrics import mols_from_smiles  # noqa: E402
from common.repro import TargetStandardizer  # noqa: E402

DATASET = "freesolv"
GEN_CKPT_PATH = TEST_ROOT / "checkpoints" / "crossmodal" / "generation" / "freesolv_mola_gen_s2025.pt"
PREDICTOR_CKPT_PATH = TEST_ROOT / "checkpoints" / "crossmodal" / "mola_graph_smiles_a2_smiles_fix" / "freesolv_mola_graph_smiles_a2_smiles_fix_s2025.pt"
PREDICTOR_HIDDEN_DIM = 256
PREDICTOR_NUM_LAYERS = 3
PREDICTOR_MAX_SM_LEN = 100
N_SAMPLES = 6
TEMPERATURE = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_CSV = TEST_ROOT / "results" / "mola" / "mola_generation_freesolv_conditioning_semantic.csv"


def load_generator():
    ckpt = torch.load(GEN_CKPT_PATH, map_location=DEVICE, weights_only=False)
    char_vocab, selfies_vocab, saved_args = ckpt["char_vocab"], ckpt["selfies_vocab"], ckpt["args"]
    mola = MoLA(
        graph_dim=30, sm_vocab_size=len(char_vocab), hidden_dim=saved_args["hidden_dim"],
        output_dim=1, num_layers=saved_args["num_layers"], positional_smiles=True,
        max_sm_len=saved_args["max_sm_len"],
    )
    model = MoLAConditionalGenerator(
        mola, vocab_size=len(selfies_vocab["token_to_id"]), hidden_dim=saved_args["hidden_dim"],
        pad_idx=selfies_vocab["pad_idx"], use_property=True, decoder_layers=saved_args["decoder_layers"],
        max_len=saved_args["max_selfies_len"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, char_vocab, selfies_vocab, saved_args


def load_independent_predictor(train_y_raw: np.ndarray, predictor_char_vocab: dict):
    """A SEPARATE, independently trained MoLA G+S regression checkpoint -- never used
    for generation. Its own state_dict has no saved args/vocab/standardizer
    (scaffold_fixed.py just torch.save()s the raw state_dict), so we deterministically
    rebuild the same char vocab (pure function of the SMILES text) and the same
    TargetStandardizer (fit on the same official train split) it was trained with."""
    standardizer = TargetStandardizer(enabled=True).fit(torch.tensor(train_y_raw, dtype=torch.float32).view(-1, 1))
    model = MoLA(
        graph_dim=30, sm_vocab_size=len(predictor_char_vocab), hidden_dim=PREDICTOR_HIDDEN_DIM,
        output_dim=1, num_layers=PREDICTOR_NUM_LAYERS, positional_smiles=True, max_sm_len=PREDICTOR_MAX_SM_LEN,
    ).to(DEVICE)
    state_dict = torch.load(PREDICTOR_CKPT_PATH, map_location=DEVICE, weights_only=False)
    result = model.load_state_dict(state_dict, strict=True)  # strict: this must be a 100% clean load, not a partial one
    assert not result.missing_keys and not result.unexpected_keys, result
    model.eval()
    return model, standardizer


def featurize_for_predictor(smiles: str, featurizer, char_vocab: dict, device):
    """Build one Data object for the independent predictor from a (generated) SMILES
    string -- same recipe crossmodal_model.data.featurize.prepare_data uses, applied to a
    single molecule."""
    graph = featurizer.featurize([smiles])[0]
    if not (hasattr(graph, "node_features") and hasattr(graph, "edge_index")):
        return None
    from torch_geometric.data import Data

    x = torch.tensor(graph.node_features, dtype=torch.float32)
    edge_index = torch.tensor(graph.edge_index, dtype=torch.long)
    sm_idx = [char_vocab.get(ch, 0) for ch in smiles[:PREDICTOR_MAX_SM_LEN]]
    if len(sm_idx) < PREDICTOR_MAX_SM_LEN:
        sm_idx.extend([0] * (PREDICTOR_MAX_SM_LEN - len(sm_idx)))
    sm = torch.tensor(sm_idx, dtype=torch.long).unsqueeze(0)
    data = Data(x=x, edge_index=edge_index, sm=sm)
    data.batch = torch.zeros(x.size(0), dtype=torch.long)
    return data.to(device)


def predict_property(model, standardizer, data) -> float:
    with torch.no_grad():
        out = model(data)[-1]  # [1,1], standardized scale
        raw = standardizer.inverse_transform(out)
    return float(raw.view(-1)[0].item())


def main() -> None:
    generator, gen_char_vocab, selfies_vocab, gen_args = load_generator()

    cfg = DATASETS[DATASET]
    csv_dir = TEST_ROOT / "data" / "deepchem_molnet" / cfg["dir"] / "csv"
    featurizer = dc.feat.MolGraphConvFeaturizer()
    train_ds = load_fixed_split(csv_dir / "train.csv", cfg["target_col"], featurizer)
    valid_ds = load_fixed_split(csv_dir / "valid.csv", cfg["target_col"], featurizer)
    test_ds = load_fixed_split(csv_dir / "test.csv", cfg["target_col"], featurizer)

    train_smiles, valid_smiles, test_smiles = list(train_ds.ids), list(valid_ds.ids), list(test_ds.ids)
    train_y_raw = np.asarray(train_ds.y, dtype="float64").reshape(-1)
    train_mean, train_std = float(train_y_raw.mean()), float(train_y_raw.std())
    print(f"Train property (raw units): mean={train_mean:.3f} std={train_std:.3f}")

    # Independent predictor's char vocab: deterministic given the same SMILES set it was
    # trained on (crossmodal_model.data.featurize.build_vocab is a pure function of the
    # character set) -- rebuilding it here reproduces the exact vocab
    # freesolv_..._a2_smiles_fix_s2025.pt used.
    predictor_char_vocab = build_char_vocab(train_smiles + valid_smiles + test_smiles)
    predictor, standardizer = load_independent_predictor(train_y_raw, predictor_char_vocab)
    print("Independent predictor loaded cleanly (strict state_dict match, 0 missing/unexpected keys).")

    train_canonical = {Chem.MolToSmiles(m) for m in mols_from_smiles(train_smiles)[0]}

    test_data = prepare_data(test_ds, np.zeros((len(test_ds.X), 0), dtype="float32"), test_smiles, gen_char_vocab, max_sm_len=gen_args["max_sm_len"])
    for d, smi in zip(test_data, test_smiles):
        d.smiles = smi

    conditions = {"-2std": -2.0 * train_std, "real": 0.0, "+2std": 2.0 * train_std}
    rows: List[Dict] = []

    print(f"\nGenerating {N_SAMPLES} samples x {len(conditions)} conditions x {len(test_data)} test molecules...")
    generator.eval()
    with torch.no_grad():
        for mi, d in enumerate(test_data):
            true_y = float(d.y.view(-1)[0].item())
            for cond_name, shift in conditions.items():
                requested = true_y + shift
                sample = d.clone().to(DEVICE)
                sample.batch = torch.zeros(sample.num_nodes, dtype=torch.long, device=DEVICE)
                prop = torch.tensor([[requested]], dtype=torch.float, device=DEVICE)
                for k in range(N_SAMPLES):
                    generated = generator.generate(sample, selfies_vocab, property_values=prop, max_len=gen_args["max_selfies_len"], temperature=TEMPERATURE, sample=True)
                    mol = Chem.MolFromSmiles(generated) if generated else None
                    canonical = Chem.MolToSmiles(mol) if mol is not None else ""
                    predicted = None
                    if mol is not None:
                        pred_data = featurize_for_predictor(canonical, featurizer, predictor_char_vocab, DEVICE)
                        if pred_data is not None:
                            predicted = predict_property(predictor, standardizer, pred_data)
                    rows.append({
                        "mol_idx": mi, "target_smiles": d.smiles, "condition": cond_name,
                        "requested_property": requested, "generated_raw": generated,
                        "generated_canonical": canonical, "valid": mol is not None,
                        "predicted_property": predicted,
                    })
            if (mi + 1) % 16 == 0:
                print(f"  ...{mi + 1}/{len(test_data)} molecules done")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved all {len(rows)} generated candidates (with predicted property) to {OUT_CSV}")

    # --- Analysis -----------------------------------------------------------------
    valid_rows = [r for r in rows if r["valid"] and r["predicted_property"] is not None]
    print(f"\n{len(valid_rows)}/{len(rows)} candidates valid AND featurizable by the independent predictor ({len(valid_rows)/len(rows):.1%})")

    requested = np.array([r["requested_property"] for r in valid_rows])
    predicted = np.array([r["predicted_property"] for r in valid_rows])
    pearson_r = float(np.corrcoef(requested, predicted)[0, 1]) if len(valid_rows) > 2 else float("nan")
    from scipy.stats import spearmanr
    spearman_r = float(spearmanr(requested, predicted).correlation) if len(valid_rows) > 2 else float("nan")
    print(f"\nRequested vs. predicted property (independent model), pooled over all valid candidates:")
    print(f"  Pearson r  = {pearson_r:.3f}")
    print(f"  Spearman r = {spearman_r:.3f}")

    print(f"\nPer-condition predicted-property distribution (independent model):")
    print(f"  {'condition':>8s} {'n_valid':>8s} {'mean':>8s} {'std':>8s} {'min':>8s} {'max':>8s}")
    for cond_name in conditions:
        vals = np.array([r["predicted_property"] for r in valid_rows if r["condition"] == cond_name])
        if len(vals):
            print(f"  {cond_name:>8s} {len(vals):8d} {vals.mean():8.3f} {vals.std():8.3f} {vals.min():8.3f} {vals.max():8.3f}")
        else:
            print(f"  {cond_name:>8s} {0:8d}  (no valid+featurizable candidates)")

    print(f"\nPer-molecule directional check (mean predicted property, +2std condition vs -2std condition):")
    n_molecules_both = 0
    n_correct_direction = 0
    for mi in range(len(test_data)):
        minus = np.array([r["predicted_property"] for r in valid_rows if r["mol_idx"] == mi and r["condition"] == "-2std"])
        plus = np.array([r["predicted_property"] for r in valid_rows if r["mol_idx"] == mi and r["condition"] == "+2std"])
        if len(minus) == 0 or len(plus) == 0:
            continue
        n_molecules_both += 1
        if plus.mean() > minus.mean():
            n_correct_direction += 1
    pct_correct = n_correct_direction / max(n_molecules_both, 1)
    print(f"  {n_correct_direction}/{n_molecules_both} molecules ({pct_correct:.1%}) have mean_predicted(+2std) > mean_predicted(-2std)")

    print(f"\nValidity / uniqueness / novelty per condition:")
    print(f"  {'condition':>8s} {'validity':>10s} {'uniqueness':>12s} {'novelty':>10s}")
    for cond_name in conditions:
        cond_rows = [r for r in rows if r["condition"] == cond_name]
        n_total = len(cond_rows)
        valid_canon = [r["generated_canonical"] for r in cond_rows if r["valid"]]
        n_valid = len(valid_canon)
        unique_canon = set(valid_canon)
        n_unique = len(unique_canon)
        n_novel = len([c for c in unique_canon if c not in train_canonical])
        validity = n_valid / max(n_total, 1)
        uniqueness = n_unique / max(n_valid, 1)
        novelty = n_novel / max(n_unique, 1)
        print(f"  {cond_name:>8s} {validity:9.1%} {uniqueness:11.1%} {novelty:9.1%}")

    print("\n=== CONCLUSION ===")
    print(f"Pearson r (requested vs. independently-predicted property) = {pearson_r:.3f}")
    print(f"Directional check (+2std > -2std): {pct_correct:.1%} of molecules")


if __name__ == "__main__":
    main()
