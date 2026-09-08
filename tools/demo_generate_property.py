"""Quick look at the conditional generator: sample a handful of molecules for a
target property value and (optionally) show the predictor's estimate for each.

    python tools/demo_generate_property.py \
        --generator-checkpoint checkpoints/gen_esol_finetune.pt \
        --predictor-checkpoint checkpoints/graph+lang_delaney_s2025.pt \
        --target-value -3.0 --num-samples 10

For the full metric suite (validity / uniqueness / novelty / diversity / FCD /
MAD) use tools/eval_generation.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from rdkit import Chem, RDLogger

from tools.eval_generation import _load_generator, _load_predictor, _predict_properties

RDLogger.DisableLog("rdApp.*")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generator-checkpoint", required=True)
    ap.add_argument("--predictor-checkpoint", default=None)
    ap.add_argument("--target-value", type=float, default=None, help="Target property value (mapped to its quantile bin)")
    ap.add_argument("--bin", type=int, default=None, help="Target bin index (alternative to --target-value)")
    ap.add_argument("--num-samples", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    gen, vocab, binner, prop_name = _load_generator(args.generator_checkpoint, args.device)
    if binner is None:
        raise ValueError("this generator was not fine-tuned conditionally (no binner in checkpoint)")

    if args.bin is not None:
        target_bin = args.bin
    elif args.target_value is not None:
        target_bin = binner.to_bin(args.target_value)
    else:
        target_bin = binner.num_bins // 2
    lo, hi = binner.bin_edges(target_bin)
    print(f"property '{prop_name}' | target bin {target_bin} -> [{lo:.3f}, {hi:.3f}] center {binner.bin_center(target_bin):.3f}")

    smiles = gen.sample(target_bin, vocab["id_to_token"], args.num_samples, max_len=args.max_len, temperature=args.temperature)
    canon = []
    for s in smiles:
        m = Chem.MolFromSmiles(s) if s else None
        canon.append(Chem.MolToSmiles(m) if m is not None else None)

    preds = None
    valid = [c for c in canon if c]
    if args.predictor_checkpoint and valid:
        model, mean, scale = _load_predictor(args.predictor_checkpoint, args.device)
        pmap = dict(zip(valid, _predict_properties(model, mean, scale, valid, args.device)))
        preds = pmap

    for i, (raw, c) in enumerate(zip(smiles, canon), 1):
        if c is None:
            print(f"{i:02d}. [invalid] {raw!r}")
        else:
            extra = f"  | predicted {prop_name}: {preds[c]:.3f}" if preds and c in preds else ""
            print(f"{i:02d}. {c}{extra}")

    n_valid = len(valid)
    print(f"\nvalid {n_valid}/{len(smiles)} | unique {len(set(valid))}/{max(n_valid, 1)}")


if __name__ == "__main__":
    main()
