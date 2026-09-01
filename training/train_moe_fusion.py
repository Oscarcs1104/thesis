"""Ablation: train the property predictor with the experimental Multi-gate
Mixture-of-Experts fusion (model/moe_fusion_model.py) instead of the main
pool-and-concatenate fusion. Predictor-only, no decoder -- for comparing Test
RMSE/NRMSE against the same dataset trained with training/train.py or
training/train_cross_attention.py.

See COMMANDS.md for usage.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data_pipeline.data import HybridGraphLangDataset, load_graph_dataset
from data_pipeline.splitters import split_dataset as split_dataset_by_strategy
from model.moe_fusion_model import build_moe_model_from_args
from training.repro import seed_everything
from training.train import (
    _target_range,
    load_graph_pretrained_checkpoint,
    load_predefined_datasets,
    resolve_predefined_split,
    run_predictor_ablation_training,
)
from training.wandb_utils import add_wandb_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the predictor with Multi-gate Mixture-of-Experts graph/language fusion")
    parser.add_argument("--data-path", type=str, default=None, help="Single CSV/graph dataset, split internally via --split. Ignored if --dataset-dir or --train-path/--val-path/--test-path is given.")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Folder with csv/train.csv, csv/valid.csv, csv/test.csv -- uses that official split as-is (D1)")
    parser.add_argument("--train-path", type=str, default=None)
    parser.add_argument("--val-path", type=str, default=None)
    parser.add_argument("--test-path", type=str, default=None)
    parser.add_argument("--output-dim", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--graph-backbone", type=str, default="gin", choices=["gcn", "gat", "gatv2", "gin"])
    parser.add_argument("--node-encoding", type=str, default="dense", choices=["categorical", "dense"])
    parser.add_argument("--node-vocab-sizes", type=int, nargs="*", default=[119, 4])
    parser.add_argument("--language-model-name", type=str, default="DeepChem/ChemBERTa-77M-MLM")
    parser.add_argument("--freeze-language-backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--num-experts", type=int, default=4, help="Shared experts in the MoE fusion layer")
    parser.add_argument("--expert-hidden-dim", type=int, default=None, help="Defaults to --hidden-dim")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--split", type=str, default="scaffold", choices=["scaffold", "random"], help="Used only when no predefined split is given")
    parser.add_argument("--standardize-target", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--lr-schedule", type=str, default="plateau", choices=["plateau", "cosine"])
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-patience", type=int, default=5)
    parser.add_argument("--min-lr-ratio", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--graph-pretrained-checkpoint", type=str, default=None, help="Load graph_encoder weights from a training/pretrain_graph.py checkpoint")
    add_wandb_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed, deterministic=args.deterministic)

    predefined = resolve_predefined_split(args)
    if predefined is not None:
        train_set, val_set, test_set = load_predefined_datasets(*predefined)
        print(f"Loaded predefined split from {predefined}: train={len(train_set)} val={len(val_set)} test={len(test_set)}")
    else:
        if not args.data_path:
            raise ValueError("Provide --dataset-dir (predefined split) or --data-path (internal split via --split).")
        dataset = HybridGraphLangDataset(load_graph_dataset(args.data_path))
        all_smiles = [str(getattr(dataset[i], "smiles", "")) for i in range(len(dataset))]
        train_set, val_set, test_set = split_dataset_by_strategy(
            dataset, args.split, args.train_ratio, args.val_ratio, args.test_ratio, args.seed, smiles_list=all_smiles
        )
    target_range = _target_range(train_set)

    criterion = nn.MSELoss()

    model = build_moe_model_from_args(args)
    if args.graph_pretrained_checkpoint:
        load_graph_pretrained_checkpoint(model, args.graph_pretrained_checkpoint)
    model = model.to(args.device)
    run_predictor_ablation_training(model, args, train_set, val_set, test_set, target_range, criterion)


if __name__ == "__main__":
    main()
