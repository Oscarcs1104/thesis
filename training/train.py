"""Minimal training entrypoint for the thesis multimodal model."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.data import Batch as GeomBatch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.data import HybridGraphLangDataset, load_graph_dataset
from model.model import build_model_from_args
from model.smiles_decoder import build_vocab as build_smiles_vocab, encode_batch as encode_smiles_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the thesis multimodal predictor")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--task", type=str, default="regression", choices=["regression", "binary"])
    parser.add_argument("--output-dim", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--graph-backbone", type=str, default="gatv2", choices=["gcn", "gat", "gatv2", "gin"])
    parser.add_argument("--language-backbone", type=str, default="huggingface", choices=["huggingface", "none"])
    parser.add_argument("--language-model-name", type=str, default="DeepChem/ChemBERTa-77M-MLM", help="Any HuggingFace text-encoder repo id (e.g. a ChemBERTa or MoLFormer checkpoint)")
    parser.add_argument("--freeze-language-backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to transformers (needed by some HF repos, e.g. MoLFormer)")
    parser.add_argument("--use-language", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--node-encoding", type=str, default="dense", choices=["categorical", "dense"])
    parser.add_argument("--node-vocab-sizes", type=int, nargs="*", default=[119, 4])
    parser.add_argument("--batch-size", type=int, default=32)
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
    parser.add_argument("--load-checkpoint", type=str, default=None)
    parser.add_argument("--graph-pretrained-checkpoint", type=str, default=None)
    parser.add_argument("--use-decoder", action="store_true")
    parser.add_argument("--training-mode", type=str, default="predictor", choices=["predictor", "joint", "decoder"])
    parser.add_argument("--decoder-loss-weight", type=float, default=0.5)
    parser.add_argument("--decoder-max-len", type=int, default=96)
    return parser.parse_args()


def split_dataset(dataset, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    total = train_ratio + val_ratio + test_ratio
    if not math.isclose(total, 1.0, rel_tol=1e-4, abs_tol=1e-4):
        raise ValueError("Ratios must sum to 1.0")
    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(dataset, [n_train, n_val, n_test], generator=generator)


def iter_graph_batches(dataset, batch_size: int, shuffle: bool):
    indices = torch.randperm(len(dataset)).tolist() if shuffle else list(range(len(dataset)))
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        batch_graphs = [dataset[idx] for idx in batch_indices]
        yield GeomBatch.from_data_list(batch_graphs)


def _dataset_target_range(dataset) -> tuple[float, float]:
    min_y = float("inf")
    max_y = -float("inf")
    for item in dataset:
        target = getattr(item, "y", None)
        if target is None:
            continue
        target_tensor = torch.as_tensor(target).float().view(-1)
        if target_tensor.numel() == 0:
            continue
        min_y = min(min_y, float(target_tensor.min().item()))
        max_y = max(max_y, float(target_tensor.max().item()))
    return (0.0, 0.0) if min_y == float("inf") else (min_y, max_y)


def evaluate(model, loader, criterion, device, task: str, target_range: tuple[float, float] | None = None, use_decoder: bool = False, training_mode: str = "predictor", decoder_vocab: dict | None = None, decoder_max_len: int = 96):
    model.eval()
    total_loss = 0.0
    total_items = 0
    correct = 0
    decoder_criterion = nn.CrossEntropyLoss(ignore_index=decoder_vocab["pad_idx"]) if use_decoder and decoder_vocab else None
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if use_decoder and training_mode == "decoder":
                smiles = getattr(batch, "smiles", None)
                if smiles is None:
                    raise ValueError("Decoder evaluation needs batch.smiles")
                decoder_inputs, decoder_targets = encode_smiles_batch(smiles, decoder_vocab, decoder_max_len, device)
                batch_y = getattr(batch, "y", None)
                property_values = (batch_y.float().unsqueeze(-1) if batch_y.dim() == 1 else batch_y.float()) if batch_y is not None else None
                _, decoder_logits = model(batch, decoder_input_ids=decoder_inputs, property_values=property_values)
                loss = decoder_criterion(decoder_logits.view(-1, decoder_logits.size(-1)), decoder_targets.view(-1))
            else:
                logits = model(batch)
                targets = batch.y.float().view_as(logits)
                loss = criterion(logits, targets)
                if task == "binary":
                    correct += ((torch.sigmoid(logits) > 0.5).float() == targets).sum().item()
            total_loss += loss.item() * batch.num_graphs
            total_items += batch.num_graphs
    metrics = {"loss": total_loss / max(total_items, 1)}
    if task == "binary" and not (use_decoder and training_mode == "decoder"):
        metrics["accuracy"] = correct / max(total_items, 1)
    elif not (use_decoder and training_mode == "decoder"):
        rmse = math.sqrt(metrics["loss"])
        metrics["rmse"] = rmse
        if target_range is not None and target_range[1] > target_range[0]:
            metrics["nrmse"] = rmse / (target_range[1] - target_range[0])
        else:
            metrics["nrmse"] = float("nan")
    return metrics


def train_one_epoch(model, loader, optimizer, criterion, device, task: str, target_range, use_decoder: bool, training_mode: str, decoder_vocab: dict | None, decoder_loss_weight: float, decoder_max_len: int):
    model.train()
    total_loss = 0.0
    total_items = 0
    decoder_criterion = nn.CrossEntropyLoss(ignore_index=decoder_vocab["pad_idx"]) if use_decoder and decoder_vocab else None
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        if use_decoder and training_mode in {"decoder", "joint"}:
            smiles = getattr(batch, "smiles", None)
            if smiles is None:
                raise ValueError("Decoder training needs batch.smiles")
            decoder_inputs, decoder_targets = encode_smiles_batch(smiles, decoder_vocab, decoder_max_len, device)
            batch_y = getattr(batch, "y", None)
            property_values = (batch_y.float().unsqueeze(-1) if batch_y.dim() == 1 else batch_y.float()) if batch_y is not None else None
            if training_mode == "decoder":
                # Unconditional pretraining is fine here (e.g. unlabeled ZINC): property_values may be None.
                _, decoder_logits = model(batch, decoder_input_ids=decoder_inputs, property_values=property_values)
                loss = decoder_criterion(decoder_logits.view(-1, decoder_logits.size(-1)), decoder_targets.view(-1))
            else:  # joint
                if batch_y is None:
                    raise ValueError("Joint training needs a labeled dataset (batch.y)")
                prop_logits, decoder_logits = model(batch, decoder_input_ids=decoder_inputs, property_values=property_values)
                prop_loss = criterion(prop_logits, batch_y.float().view_as(prop_logits))
                dec_loss = decoder_criterion(decoder_logits.view(-1, decoder_logits.size(-1)), decoder_targets.view(-1))
                loss = prop_loss + decoder_loss_weight * dec_loss
        else:
            logits = model(batch)
            targets = batch.y.float().view_as(logits)
            loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        total_items += batch.num_graphs
    metrics = {"loss": total_loss / max(total_items, 1)}
    if task != "binary":
        rmse = math.sqrt(metrics["loss"])
        metrics["rmse"] = rmse
        if target_range is not None and target_range[1] > target_range[0]:
            metrics["nrmse"] = rmse / (target_range[1] - target_range[0])
        else:
            metrics["nrmse"] = float("nan")
    return metrics


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    dataset = load_graph_dataset(args.data_path)
    dataset = HybridGraphLangDataset(dataset)
    train_set, val_set, test_set = split_dataset(dataset, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)

    decoder_vocab = None
    if args.use_decoder:
        train_smiles = [str(getattr(train_set[idx], "smiles", "")) for idx in range(len(train_set))]
        decoder_vocab = build_smiles_vocab(train_smiles)
        args.decoder_vocab_size = len(decoder_vocab["token_to_id"])
        args.decoder_pad_idx = decoder_vocab["pad_idx"]
        args.decoder_start_idx = decoder_vocab["start_idx"]
        args.decoder_end_idx = decoder_vocab["end_idx"]

    if args.task == "binary":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()

    model = build_model_from_args(args).to(args.device)
    if args.load_checkpoint:
        ckpt = torch.load(args.load_checkpoint, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict"))
        if state_dict is None:
            raise ValueError("Checkpoint does not contain model_state_dict or state_dict")
        model.load_state_dict(state_dict, strict=False)
    if args.graph_pretrained_checkpoint:
        checkpoint = torch.load(args.graph_pretrained_checkpoint, map_location="cpu")
        state_dict = checkpoint.get("graph_encoder", checkpoint.get("encoder_state_dict", checkpoint.get("model_state_dict", {})))
        if isinstance(state_dict, dict):
            model.graph_encoder.load_state_dict(state_dict, strict=False)

    if args.use_decoder and args.training_mode == "decoder":
        sample_graph = train_set[0]
        if not hasattr(sample_graph, "batch") or sample_graph.batch is None:
            sample_graph.batch = torch.zeros(sample_graph.num_nodes, dtype=torch.long)
        sample_graph = sample_graph.to(args.device)
        with torch.no_grad():
            _ = model(
                sample_graph,
                decoder_input_ids=torch.zeros(1, 1, dtype=torch.long, device=args.device),
                property_values=torch.zeros(1, 1, dtype=torch.float, device=args.device),
            )
        for name, parameter in model.named_parameters():
            if name.startswith("decoder") or name.startswith("decoder_condition_proj"):
                parameter.requires_grad_(True)
            else:
                parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    target_range = _dataset_target_range(dataset)

    best_val = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_loader = iter_graph_batches(train_set, args.batch_size, shuffle=True)
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, args.device, args.task, target_range, args.use_decoder, args.training_mode, decoder_vocab, args.decoder_loss_weight, args.decoder_max_len)
        val_loader = iter_graph_batches(val_set, args.batch_size, shuffle=False)
        val_metrics = evaluate(model, val_loader, criterion, args.device, args.task, target_range, use_decoder=args.use_decoder, training_mode=args.training_mode, decoder_vocab=decoder_vocab, decoder_max_len=args.decoder_max_len)
        if args.use_decoder and args.training_mode == "decoder":
            print(f"Epoch {epoch:03d} | train_decoder_loss={train_metrics['loss']:.4f} | val_decoder_loss={val_metrics['loss']:.4f}")
        else:
            print(f"Epoch {epoch:03d} | train_loss={train_metrics['loss']:.4f} | val_loss={val_metrics['loss']:.4f}")
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "args": vars(args),
                "decoder_vocab": decoder_vocab,
            }
            checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else Path("checkpoints") / "best.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, checkpoint_path)
        if epoch > args.patience and val_metrics["loss"] >= best_val:
            break

    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])
    test_loader = iter_graph_batches(test_set, args.batch_size, shuffle=False)
    test_metrics = evaluate(model, test_loader, criterion, args.device, args.task, target_range, use_decoder=args.use_decoder, training_mode=args.training_mode, decoder_vocab=decoder_vocab, decoder_max_len=args.decoder_max_len)
    if args.use_decoder and args.training_mode == "decoder":
        print(f"Test decoder loss={test_metrics['loss']:.4f}")
    else:
        print(f"Test loss={test_metrics['loss']:.4f}")


if __name__ == "__main__":
    main()
