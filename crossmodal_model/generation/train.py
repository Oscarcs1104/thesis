"""Teacher-forcing training for MoLAConditionalGenerator (see decoder.py):
property-conditioned molecule generation, decoder cross-attends over the shared MoLA
encoder's raw graph-node + SMILES-char states (Chemformer-style), SELFIES vocabulary/
target (common/selfies_vocab.py).

Reuses crossmodal_model/train/core.py's data loading (same official scaffold-split CSVs,
same MolGraphConvFeaturizer) so this is directly comparable in data terms to the
property-prediction benchmarks already run.

Evaluation: teacher-forcing val/test loss + token accuracy, PLUS a real generative
eval -- SAMPLING (not just greedy) --num-samples-per-mol candidates per test molecule
(conditioned on that molecule's own true property value), scored with
common/mol_metrics.py's primitives for validity/uniqueness/novelty-vs-train. All
generated candidates are saved to --samples-out for later inspection.

Usage:
    python crossmodal_model/generation/train.py --dataset freesolv --epochs 100 --patience 15
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import warnings
from pathlib import Path
from typing import List

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

THIS_DIR = Path(__file__).resolve().parent
TEST_ROOT = THIS_DIR.parent.parent
if str(TEST_ROOT) not in sys.path:
    sys.path.append(str(TEST_ROOT))

import deepchem as dc  # noqa: E402
from torch_geometric.loader import DataLoader as GeomDataLoader  # noqa: E402

from crossmodal_model.data.featurize import build_vocab as build_char_vocab, prepare_data  # noqa: E402
from crossmodal_model.generation.decoder import MoLAConditionalGenerator, build_selfies_vocab, encode_batch  # noqa: E402
from crossmodal_model.model.mola import MoLA  # noqa: E402
from crossmodal_model.train.core import DATASETS, load_fixed_split  # noqa: E402
from common.mol_metrics import mols_from_smiles  # noqa: E402
from common.repro import seed_everything  # noqa: E402
from common.wandb_utils import add_wandb_args, wandb_finish, wandb_init, wandb_log  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MoLA's property-conditioned generation decoder")
    parser.add_argument("--dataset", type=str, default="freesolv", choices=list(DATASETS.keys()))
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3, help="MoLA encoder layers")
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--max-sm-len", type=int, default=100, help="Encoder-side (char-level) SMILES input length")
    parser.add_argument("--max-selfies-len", type=int, default=100, help="Decoder-side (SELFIES) target length")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no-property", dest="use_property", action="store_false", help="Unconditional (autoencoder) generation instead of property-conditioned")
    parser.add_argument("--num-samples-per-mol", type=int, default=5, help="Sampled (not greedy) candidates per test molecule for the validity/uniqueness/novelty eval")
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    parser.add_argument("--samples-out", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    add_wandb_args(parser)
    parser.set_defaults(wandb_project="mola-conditional-generation")  # new project, not thesis-multimodal
    args = parser.parse_args()
    if args.samples_out is None:
        args.samples_out = str(TEST_ROOT / "results" / "mola" / f"mola_generation_{args.dataset}_samples.csv")
    return args


def build_data(dataset_name: str, max_sm_len: int):
    cfg = DATASETS[dataset_name]
    csv_dir = TEST_ROOT / "data" / "deepchem_molnet" / cfg["dir"] / "csv"
    featurizer = dc.feat.MolGraphConvFeaturizer()

    def _load(split):
        ds = load_fixed_split(csv_dir / f"{split}.csv", cfg["target_col"], featurizer)
        smiles = list(ds.ids)
        zero_molformer = __import__("numpy").zeros((len(ds.X), 0), dtype="float32")
        data_list = prepare_data(ds, zero_molformer, smiles, char_vocab, max_sm_len=max_sm_len)
        for d, smi in zip(data_list, smiles):
            d.smiles = smi  # not set by prepare_data -- needed as the decoder's teacher-forcing target
        return data_list, smiles

    train_smiles_raw = list(load_fixed_split(csv_dir / "train.csv", cfg["target_col"], featurizer).ids)
    valid_smiles_raw = list(load_fixed_split(csv_dir / "valid.csv", cfg["target_col"], featurizer).ids)
    test_smiles_raw = list(load_fixed_split(csv_dir / "test.csv", cfg["target_col"], featurizer).ids)
    char_vocab = build_char_vocab(train_smiles_raw + valid_smiles_raw + test_smiles_raw)  # encoder-side, unchanged

    train_data, train_smiles = _load("train")
    valid_data, valid_smiles = _load("valid")
    test_data, test_smiles = _load("test")
    return train_data, valid_data, test_data, char_vocab, train_smiles + valid_smiles + test_smiles


def main() -> None:
    args = parse_args()
    seed_everything(args.seed, deterministic=False)
    device = args.device

    run_name = args.wandb_run_name or f"{args.dataset}-mola-gen-s{args.seed}"
    wandb_run = wandb_init(argparse.Namespace(**{**vars(args), "wandb_run_name": run_name}), config=vars(args))

    print(f"Loading {args.dataset} (official scaffold split)...")
    train_data, valid_data, test_data, char_vocab, all_smiles = build_data(args.dataset, args.max_sm_len)
    print(f"  sizes: train={len(train_data)} valid={len(valid_data)} test={len(test_data)}")

    selfies_vocab = build_selfies_vocab(all_smiles)  # decoder target vocab -- SELFIES tokens, from ALL splits' SMILES text
    print(f"  char vocab (encoder input) size={len(char_vocab)} | SELFIES vocab (decoder target) size={len(selfies_vocab['token_to_id'])}")

    mola = MoLA(
        graph_dim=train_data[0].x.size(1),
        sm_vocab_size=len(char_vocab),
        hidden_dim=args.hidden_dim,
        output_dim=1,
        num_layers=args.num_layers,
        positional_smiles=True,
        max_sm_len=args.max_sm_len,
    )
    model = MoLAConditionalGenerator(
        mola,
        vocab_size=len(selfies_vocab["token_to_id"]),
        hidden_dim=args.hidden_dim,
        pad_idx=selfies_vocab["pad_idx"],
        use_property=args.use_property,
        decoder_layers=args.decoder_layers,
        max_len=args.max_selfies_len,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=selfies_vocab["pad_idx"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    train_loader = GeomDataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = GeomDataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
    test_loader = GeomDataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    def run_epoch(loader, train: bool):
        model.train(train)
        total_loss, total_correct, total_tokens, total_items = 0.0, 0, 0, 0
        for batch in loader:
            batch = batch.to(device)
            decoder_inputs, decoder_targets = encode_batch(batch.smiles, selfies_vocab, args.max_selfies_len, device)
            property_values = batch.y.float().view(-1, 1) if args.use_property else None

            logits = model(batch, decoder_inputs, property_values=property_values)
            loss = criterion(logits.reshape(-1, logits.size(-1)), decoder_targets.reshape(-1))

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            mask = decoder_targets != selfies_vocab["pad_idx"]
            preds = logits.argmax(dim=-1)
            total_correct += int(((preds == decoder_targets) & mask).sum().item())
            total_tokens += int(mask.sum().item())
            total_loss += loss.item() * batch.num_graphs
            total_items += batch.num_graphs
        return total_loss / max(total_items, 1), total_correct / max(total_tokens, 1)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else TEST_ROOT / "checkpoints" / "crossmodal" / "generation" / f"{args.dataset}_mola_gen_s{args.seed}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(train_loader, train=True)
        with torch.no_grad():
            val_loss, val_acc = run_epoch(valid_loader, train=False)
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} acc={train_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f} | {time.time()-start:.1f}s")
        wandb_log(wandb_run, {"train/loss": train_loss, "train/token_acc": train_acc, "val/loss": val_loss, "val/token_acc": val_acc}, step=epoch)

        if val_loss < best_val_loss:
            best_val_loss, epochs_without_improvement = val_loss, 0
            torch.save({"model_state_dict": model.state_dict(), "selfies_vocab": selfies_vocab, "char_vocab": char_vocab, "args": vars(args)}, checkpoint_path)
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            print(f"Early stop at epoch {epoch}")
            break

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded best checkpoint (val_loss={best_val_loss:.4f}) from {checkpoint_path}")
    with torch.no_grad():
        test_loss, test_acc = run_epoch(test_loader, train=False)
    print(f"Test loss={test_loss:.4f} token_acc={test_acc:.4f}")
    wandb_log(wandb_run, {"test/loss": test_loss, "test/token_acc": test_acc})

    print(f"\nGenerative eval: sampling {args.num_samples_per_mol} candidates/molecule (T={args.sample_temperature}) over {len(test_data)} test molecules...")
    gen_metrics = sample_and_evaluate(
        model, test_data, selfies_vocab,
        train_smiles=[d.smiles for d in train_data],
        device=device, num_samples=args.num_samples_per_mol, temperature=args.sample_temperature,
        use_property=args.use_property, max_len=args.max_selfies_len, out_csv=Path(args.samples_out),
    )
    print(
        f"Generated={gen_metrics['num_generated']} | Valid={gen_metrics['num_valid']} ({gen_metrics['validity']:.1%}) "
        f"| Unique(of valid)={gen_metrics['num_unique']} ({gen_metrics['uniqueness']:.1%}) "
        f"| Novel(of unique, vs train)={gen_metrics['num_novel']} ({gen_metrics['novelty']:.1%})"
    )
    print(f"Saved all generated candidates to {args.samples_out}")

    print("\n=== SUMMARY ===")
    print(f"best_val_loss={best_val_loss:.4f} test_loss={test_loss:.4f} test_token_acc={test_acc:.4f} "
          f"validity={gen_metrics['validity']:.4f} uniqueness={gen_metrics['uniqueness']:.4f} novelty={gen_metrics['novelty']:.4f}")

    wandb_log(wandb_run, {
        "gen/num_generated": gen_metrics["num_generated"], "gen/num_valid": gen_metrics["num_valid"],
        "gen/num_unique": gen_metrics["num_unique"], "gen/num_novel": gen_metrics["num_novel"],
        "gen/validity": gen_metrics["validity"], "gen/uniqueness": gen_metrics["uniqueness"],
        "gen/novelty": gen_metrics["novelty"], "best_val_loss": best_val_loss,
    })
    if wandb_run is not None:
        import wandb as _wandb

        with open(args.samples_out, newline="", encoding="utf-8") as handle:
            sample_reader = csv.reader(handle)
            header = next(sample_reader)
            rows_preview = [row for _, row in zip(range(50), sample_reader)]  # first 50 rows -- quick inspection in the UI
        wandb_run.log({"gen/samples_preview": _wandb.Table(columns=header, data=rows_preview)})
    wandb_finish(wandb_run)


def sample_and_evaluate(model, test_data, vocab, train_smiles: List[str], device, num_samples: int, temperature: float, use_property: bool, max_len: int, out_csv: Path) -> dict:
    """Property-conditioned SAMPLING (not greedy) generative eval: validity (RDKit-parseable),
    uniqueness (distinct canonical SMILES among the valid ones), novelty (of those, not an
    exact canonical match to any TRAIN molecule). Saves every candidate (valid or not) to
    out_csv for later inspection."""
    model.eval()

    train_mols, _ = mols_from_smiles(train_smiles)
    train_canonical = {Chem.MolToSmiles(m) for m in train_mols}

    rows = []
    valid_canonical: List[str] = []
    with torch.no_grad():
        for d in test_data:
            sample = d.clone().to(device)
            sample.batch = torch.zeros(sample.num_nodes, dtype=torch.long, device=device)
            prop = sample.y.float().view(1, 1) if use_property else None
            for k in range(num_samples):
                generated = model.generate(sample, vocab, property_values=prop, max_len=max_len, temperature=temperature, sample=True)
                mol = Chem.MolFromSmiles(generated) if generated else None
                canonical = Chem.MolToSmiles(mol) if mol is not None else ""
                rows.append({
                    "target_smiles": d.smiles,
                    "target_property": float(d.y.view(-1)[0].item()),
                    "sample_idx": k,
                    "generated_raw": generated,
                    "generated_canonical": canonical,
                    "valid": mol is not None,
                })
                if mol is not None:
                    valid_canonical.append(canonical)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["target_smiles", "target_property", "sample_idx", "generated_raw", "generated_canonical", "valid"])
        writer.writeheader()
        writer.writerows(rows)

    num_generated = len(rows)
    num_valid = len(valid_canonical)
    unique_canonical = set(valid_canonical)
    num_unique = len(unique_canonical)
    novel = [c for c in unique_canonical if c not in train_canonical]
    num_novel = len(novel)

    return {
        "num_generated": num_generated,
        "num_valid": num_valid,
        "num_unique": num_unique,
        "num_novel": num_novel,
        "validity": num_valid / max(num_generated, 1),
        "uniqueness": num_unique / max(num_valid, 1),
        "novelty": num_novel / max(num_unique, 1),
    }


if __name__ == "__main__":
    main()
