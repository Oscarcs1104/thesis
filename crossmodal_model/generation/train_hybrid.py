"""Joint-loss (regression + generation) training on HybridMoLA: one shared encoder, per
batch loss = reg_loss + --gen-loss-weight * gen_loss, one optimizer step (covers
HybridMoLA's regression head too, since MoLAConditionalGenerator holds `self.mola = mola`
as a submodule). Reconstruction-style conditioning (property = seed's own true value).

Usage:
    python crossmodal_model/generation/train_hybrid.py --dataset freesolv --epochs 100 --patience 15
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd
import torch
import torch.nn as nn
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

THIS_DIR = Path(__file__).resolve().parent
TEST_ROOT = THIS_DIR.parent.parent
if str(TEST_ROOT) not in sys.path:
    sys.path.append(str(TEST_ROOT))

from torch_geometric.loader import DataLoader as GeomDataLoader  # noqa: E402

from crossmodal_model.data.featurize import build_vocab  # noqa: E402
from crossmodal_model.data.featurize_hybrid import prepare_hybrid_data  # noqa: E402
from crossmodal_model.generation.decoder import MoLAConditionalGenerator, build_selfies_vocab, encode_batch  # noqa: E402
from crossmodal_model.generation.train import sample_and_evaluate  # noqa: E402
from crossmodal_model.model.mola_hybrid import HybridMoLA  # noqa: E402
from crossmodal_model.train.core import DATASETS  # noqa: E402
from common.repro import TargetStandardizer, build_scheduler, regression_metrics, seed_everything, step_scheduler  # noqa: E402
from common.wandb_utils import add_wandb_args, wandb_finish, wandb_init, wandb_log  # noqa: E402

CSV_FIELDS = [
    "dataset", "config", "seed", "reg_loss", "rmse", "nrmse", "mae", "mse", "r2",
    "gen_val_loss", "token_acc", "validity", "uniqueness", "novelty", "elapsed_s",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joint regression+generation loss on HybridMoLA (thesis_model graph branch)")
    parser.add_argument("--dataset", type=str, default="freesolv", choices=list(DATASETS.keys()))
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--max-sm-len", type=int, default=100)
    parser.add_argument("--max-selfies-len", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-patience", type=int, default=5)
    parser.add_argument("--min-lr-ratio", type=float, default=0.01)
    parser.add_argument("--gen-loss-weight", type=float, default=1.0)
    parser.add_argument("--num-samples-per-mol", type=int, default=5)
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    parser.add_argument("--samples-out", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    add_wandb_args(parser)
    parser.set_defaults(wandb_project="mola-conditional-generation")
    args = parser.parse_args()
    if args.samples_out is None:
        args.samples_out = str(TEST_ROOT / "results" / "mola" / f"mola_hybrid_gen_{args.dataset}_samples.csv")
    if args.out is None:
        args.out = str(TEST_ROOT / "results" / "mola" / "NewTestHybridJoint_benchmark.csv")
    if args.checkpoint_path is None:
        args.checkpoint_path = str(TEST_ROOT / "checkpoints" / "crossmodal" / "hybrid_joint" / f"{args.dataset}_mola_hybrid_joint_s{args.seed}.pt")
    return args


def build_data(dataset_name: str, max_sm_len: int):
    cfg = DATASETS[dataset_name]
    csv_dir = TEST_ROOT / "data" / "deepchem_molnet" / cfg["dir"] / "csv"

    def _load(split):
        df = pd.read_csv(csv_dir / f"{split}.csv")
        smiles = df["smiles"].astype(str).tolist()
        y = df[cfg["target_col"]].astype(float).tolist()
        return smiles, y

    train_smiles, train_y = _load("train")
    valid_smiles, valid_y = _load("valid")
    test_smiles, test_y = _load("test")
    char_vocab = build_vocab(train_smiles + valid_smiles + test_smiles)

    train_data = prepare_hybrid_data(train_smiles, train_y, char_vocab, max_sm_len=max_sm_len)
    valid_data = prepare_hybrid_data(valid_smiles, valid_y, char_vocab, max_sm_len=max_sm_len)
    test_data = prepare_hybrid_data(test_smiles, test_y, char_vocab, max_sm_len=max_sm_len)
    all_smiles = [d.smiles for d in (train_data + valid_data + test_data)]
    return train_data, valid_data, test_data, char_vocab, all_smiles


def main() -> None:
    args = parse_args()
    seed_everything(args.seed, deterministic=False)
    device = args.device
    start = time.time()

    run_name = args.wandb_run_name or f"{args.dataset}-mola-hybrid-gen-s{args.seed}"
    wandb_run = wandb_init(argparse.Namespace(**{**vars(args), "wandb_run_name": run_name}), config=vars(args))

    print(f"Loading {args.dataset} (official scaffold split, hybrid/categorical featurization)...")
    train_data, valid_data, test_data, char_vocab, all_smiles = build_data(args.dataset, args.max_sm_len)
    print(f"  sizes: train={len(train_data)} valid={len(valid_data)} test={len(test_data)}")
    selfies_vocab = build_selfies_vocab(all_smiles)
    print(f"  char vocab (encoder input) size={len(char_vocab)} | SELFIES vocab (decoder target) size={len(selfies_vocab['token_to_id'])}")

    train_y = torch.stack([d.y.float().view(-1) for d in train_data])
    standardizer = TargetStandardizer(enabled=True).fit(train_y)
    target_std = float(train_y.std())

    mola = HybridMoLA(
        sm_vocab_size=len(char_vocab), hidden_dim=args.hidden_dim, output_dim=1,
        num_layers=args.num_layers, positional_smiles=True, max_sm_len=args.max_sm_len,
    )
    model = MoLAConditionalGenerator(
        mola, vocab_size=len(selfies_vocab["token_to_id"]), hidden_dim=args.hidden_dim,
        pad_idx=selfies_vocab["pad_idx"], use_property=True, decoder_layers=args.decoder_layers,
        max_len=args.max_selfies_len,
    ).to(device)

    reg_criterion = nn.MSELoss()
    gen_criterion = nn.CrossEntropyLoss(ignore_index=selfies_vocab["pad_idx"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler("plateau", optimizer, args.warmup_epochs, args.epochs, args.min_lr_ratio, args.plateau_factor, args.plateau_patience)

    train_loader = GeomDataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = GeomDataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
    test_loader = GeomDataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    def run_epoch(loader, train: bool) -> dict:
        model.train(train)
        total_reg_loss = total_gen_loss = total_combined = 0.0
        total_correct = total_tokens = total_items = 0
        all_preds, all_targets = [], []
        for batch in loader:
            batch = batch.to(device)

            reg_out = model.mola(batch)[-1]
            targets_orig = batch.y.float().view_as(reg_out)
            targets_std = standardizer.transform(targets_orig)
            reg_loss = reg_criterion(reg_out, targets_std)

            decoder_inputs, decoder_targets = encode_batch(batch.smiles, selfies_vocab, args.max_selfies_len, device)
            property_values = batch.y.float().view(-1, 1)
            gen_logits = model(batch, decoder_inputs, property_values=property_values)
            gen_loss = gen_criterion(gen_logits.reshape(-1, gen_logits.size(-1)), decoder_targets.reshape(-1))

            combined = reg_loss + args.gen_loss_weight * gen_loss

            if train:
                optimizer.zero_grad(set_to_none=True)
                combined.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optimizer.step()

            preds_orig = standardizer.inverse_transform(reg_out)
            all_preds.append(preds_orig.detach().cpu())
            all_targets.append(targets_orig.detach().cpu())

            mask = decoder_targets != selfies_vocab["pad_idx"]
            preds_tok = gen_logits.argmax(dim=-1)
            total_correct += int(((preds_tok == decoder_targets) & mask).sum().item())
            total_tokens += int(mask.sum().item())

            total_reg_loss += reg_loss.item() * batch.num_graphs
            total_gen_loss += gen_loss.item() * batch.num_graphs
            total_combined += combined.item() * batch.num_graphs
            total_items += batch.num_graphs

        reg_metrics = regression_metrics(torch.cat(all_preds), torch.cat(all_targets), target_std)
        return {
            "reg_loss": total_reg_loss / max(total_items, 1),
            "gen_loss": total_gen_loss / max(total_items, 1),
            "combined_loss": total_combined / max(total_items, 1),
            "token_acc": total_correct / max(total_tokens, 1),
            **{f"reg_{k}": v for k, v in reg_metrics.items()},
        }

    best_val_combined = float("inf")
    epochs_without_improvement = 0
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(train_loader, train=True)
        with torch.no_grad():
            val_metrics = run_epoch(valid_loader, train=False)
        step_scheduler(scheduler, val_metrics["combined_loss"])
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d} | lr={lr_now:.2e} | "
            f"train: reg={train_metrics['reg_loss']:.4f} gen={train_metrics['gen_loss']:.4f} acc={train_metrics['token_acc']:.3f} | "
            f"val: reg={val_metrics['reg_loss']:.4f} rmse={val_metrics.get('reg_rmse', float('nan')):.4f} "
            f"gen={val_metrics['gen_loss']:.4f} acc={val_metrics['token_acc']:.3f}"
        )
        wandb_log(wandb_run, {**{f"train/{k}": v for k, v in train_metrics.items()}, **{f"val/{k}": v for k, v in val_metrics.items()}, "lr": lr_now}, step=epoch)

        if val_metrics["combined_loss"] < best_val_combined:
            best_val_combined = val_metrics["combined_loss"]
            epochs_without_improvement = 0
            torch.save({"model_state_dict": model.state_dict(), "selfies_vocab": selfies_vocab, "char_vocab": char_vocab, "args": vars(args)}, checkpoint_path)
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            print(f"Early stop at epoch {epoch}")
            break

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded best checkpoint (val_combined_loss={best_val_combined:.4f}) from {checkpoint_path}")

    with torch.no_grad():
        test_metrics = run_epoch(test_loader, train=False)
    print(
        f"Test: reg_loss={test_metrics['reg_loss']:.4f} rmse={test_metrics.get('reg_rmse', float('nan')):.4f} "
        f"r2={test_metrics.get('reg_r2', float('nan')):.4f} | gen_loss={test_metrics['gen_loss']:.4f} token_acc={test_metrics['token_acc']:.4f}"
    )

    print(f"\nGenerative eval: sampling {args.num_samples_per_mol} candidates/molecule...")
    gen_eval = sample_and_evaluate(
        model, test_data, selfies_vocab, train_smiles=[d.smiles for d in train_data],
        device=device, num_samples=args.num_samples_per_mol, temperature=args.sample_temperature,
        use_property=True, max_len=args.max_selfies_len, out_csv=Path(args.samples_out),
    )
    print(f"Validity={gen_eval['validity']:.1%} Uniqueness={gen_eval['uniqueness']:.1%} Novelty={gen_eval['novelty']:.1%}")

    elapsed = time.time() - start
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()
    with out_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "dataset": args.dataset, "config": "NewTestHybridJoint", "seed": args.seed,
            "reg_loss": test_metrics["reg_loss"], "rmse": test_metrics.get("reg_rmse"),
            "nrmse": test_metrics.get("reg_nrmse"), "mae": test_metrics.get("reg_mae"),
            "mse": test_metrics.get("reg_mse"), "r2": test_metrics.get("reg_r2"),
            "gen_val_loss": test_metrics["gen_loss"], "token_acc": test_metrics["token_acc"],
            "validity": gen_eval["validity"], "uniqueness": gen_eval["uniqueness"], "novelty": gen_eval["novelty"],
            "elapsed_s": elapsed,
        })
    print(f"Wrote {out_path}")

    wandb_log(wandb_run, {**{f"test/{k}": v for k, v in test_metrics.items()}, **{f"gen/{k}": v for k, v in gen_eval.items()}})
    wandb_finish(wandb_run)


if __name__ == "__main__":
    main()
