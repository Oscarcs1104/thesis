"""Ablation: train the property predictor with the experimental cross-attention
fusion (model/cross_attention_model.py) instead of the main pool-and-concatenate
fusion. Predictor-only, no decoder -- for comparing Test RMSE/NRMSE against the
same dataset trained with training/train.py.

Usage:
  python training/train_cross_attention.py --data-path data/esol.csv --epochs 50 --device cuda
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
from model.cross_attention_model import build_cross_attention_model_from_args
from training.train import _dataset_target_range, run_predictor_ablation_training, split_dataset
from training.wandb_utils import add_wandb_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the predictor with cross-attention graph/language fusion")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--task", type=str, default="regression", choices=["regression", "binary"])
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
    parser.add_argument("--num-heads", type=int, default=4, help="Attention heads in each cross-attention layer")
    parser.add_argument("--num-cross-layers", type=int, default=1, help="How many rounds of graph<->language cross-attention")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    add_wandb_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    dataset = load_graph_dataset(args.data_path)
    dataset = HybridGraphLangDataset(dataset)
    train_set, val_set, test_set = split_dataset(dataset, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
    target_range = _dataset_target_range(dataset)

    criterion = nn.BCEWithLogitsLoss() if args.task == "binary" else nn.MSELoss()

    model = build_cross_attention_model_from_args(args).to(args.device)
    run_predictor_ablation_training(model, args, train_set, val_set, test_set, target_range, criterion)


if __name__ == "__main__":
    main()
