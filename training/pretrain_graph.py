"""Contrastive pretraining for the graph encoder (produces a checkpoint usable via
train.py's --graph-pretrained-checkpoint)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch as GeomBatch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data_pipeline.data import load_graph_dataset
from model.encoders import GraphEncoder
from training.wandb_utils import add_wandb_args, wandb_finish, wandb_init, wandb_log


def _print_progress(prefix: str, current: int, total: int, start_time: float) -> None:
    total = max(total, 1)
    width = 24
    filled = int(width * current / total)
    bar = "=" * filled + "." * (width - filled)
    elapsed = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed, 60)
    hours, minutes = divmod(minutes, 60)
    clock = f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"
    print(f"{prefix} [{bar}] {100.0 * current / total:5.1f}% | elapsed {clock}", end="\r", flush=True)


def _augment_view(graph):
    view = graph.clone()
    if getattr(view, "edge_index", None) is not None and view.edge_index.numel() > 0:
        mask = torch.rand(view.edge_index.size(1)) >= 0.15
        view.edge_index = view.edge_index[:, mask]
    if getattr(view, "x", None) is not None:
        view.x = view.x + torch.randn_like(view.x) * 0.01
    return view


class ContrastiveGraphDataset(Dataset):
    """Produces two augmented views per graph; the actual augmentation work happens
    inside __getitem__ so DataLoader workers can parallelize it across CPUs."""

    def __init__(self, graphs) -> None:
        self.graphs = graphs

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int):
        graph = self.graphs[idx]
        return _augment_view(graph), _augment_view(graph)


def _collate_views(pairs):
    view_1, view_2 = zip(*pairs)
    return GeomBatch.from_data_list(list(view_1)), GeomBatch.from_data_list(list(view_2))


class GraphPretrainer:
    def __init__(self, encoder: GraphEncoder, hidden_dim: int, proj_dim: int, device: str) -> None:
        self.encoder = encoder.to(device)
        self.device = torch.device(device)
        self.proj = nn.Sequential(nn.Linear(hidden_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim)).to(self.device)
        self.opt = torch.optim.AdamW(list(self.encoder.parameters()) + list(self.proj.parameters()), lr=1e-4)

    def _nt_xent(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        reps = torch.cat([z1, z2], dim=0)
        sim = reps @ reps.t() / 0.1
        labels = torch.arange(z1.size(0), device=self.device)
        targets = torch.cat([labels + z1.size(0), labels], dim=0)
        mask = torch.eye(2 * z1.size(0), dtype=torch.bool, device=self.device)
        sim.masked_fill_(mask, -9e15)
        return F.cross_entropy(sim, targets)

    def train_epoch(self, loader: DataLoader, on_batch_end=None) -> float:
        self.encoder.train()
        total_loss = 0.0
        total_batches = len(loader)
        for step, (view_1, view_2) in enumerate(loader, start=1):
            view_1 = view_1.to(self.device)
            view_2 = view_2.to(self.device)
            _, states_1 = self.encoder(view_1.x, view_1.edge_index, view_1.batch)
            _, states_2 = self.encoder(view_2.x, view_2.edge_index, view_2.batch)
            loss = self._nt_xent(self.proj(states_1[-1]), self.proj(states_2[-1]))
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
            total_loss += loss.item()
            if on_batch_end is not None:
                on_batch_end(step, total_batches)
        return total_loss / max(total_batches, 1)


def train_graph(
    graphs,
    out: str,
    epochs: int = 5,
    hidden_dim: int = 256,
    num_layers: int = 3,
    dropout: float = 0.3,
    graph_backbone: str = "gatv2",
    device: str = "cpu",
    batch_size: int = 64,
    node_encoding: str = "dense",
    num_workers: int = 0,
    wandb_run=None,
) -> None:
    encoder = GraphEncoder(hidden_dim=hidden_dim, graph_backbone=graph_backbone, num_layers=num_layers, dropout=dropout, node_encoding=node_encoding)
    trainer = GraphPretrainer(encoder=encoder, hidden_dim=hidden_dim, proj_dim=128, device=device)
    loader = DataLoader(
        ContrastiveGraphDataset(graphs),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        collate_fn=_collate_views,
    )
    for epoch in range(1, epochs + 1):
        start = time.time()
        loss = trainer.train_epoch(loader, on_batch_end=lambda current, total: _print_progress(f"Graph epoch {epoch}/{epochs}", current, total, start))
        _print_progress(f"Graph epoch {epoch}/{epochs}", 1, 1, start)
        print(f" | loss={loss:.4f}")
        wandb_log(wandb_run, {"train/loss": loss}, step=epoch)
    torch.save({"stage": "graph", "graph_encoder": encoder.state_dict()}, out)
    print(f"Saved graph pretrain to {out}")
    wandb_finish(wandb_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Contrastive pretraining for the graph encoder")
    parser.add_argument("--data-path", type=str, required=True, help="Graph dataset path (see data_pipeline.data.load_graph_dataset)")
    parser.add_argument("--out", type=str, default="graph_pretrain.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0, help="Parallel CPU processes for batch loading (e.g. 4 if your machine has 4 CPUs)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--graph-backbone", type=str, default="gatv2", choices=["gcn", "gat", "gatv2", "gin"])
    parser.add_argument("--node-encoding", type=str, default="dense", choices=["categorical", "dense"])
    add_wandb_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graphs = load_graph_dataset(args.data_path)
    wandb_run = wandb_init(args, config=vars(args))
    train_graph(
        graphs,
        args.out,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        graph_backbone=args.graph_backbone,
        device=args.device,
        batch_size=args.batch_size,
        node_encoding=args.node_encoding,
        num_workers=args.num_workers,
        wandb_run=wandb_run,
    )


if __name__ == "__main__":
    main()
