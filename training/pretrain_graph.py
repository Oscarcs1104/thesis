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
from torch_geometric.data import Batch as GeomBatch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.data import load_graph_dataset
from model.encoders import GraphEncoder


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


def _graph_views(graphs, device: torch.device):
    batch_graphs = [g.clone() for g in graphs]
    view_1 = []
    view_2 = []
    for graph in batch_graphs:
        g1 = graph.clone()
        g2 = graph.clone()
        if getattr(g1, "edge_index", None) is not None and g1.edge_index.numel() > 0:
            mask = torch.rand(g1.edge_index.size(1)) >= 0.15
            g1.edge_index = g1.edge_index[:, mask]
            g2.edge_index = g2.edge_index[:, mask]
        if getattr(g1, "x", None) is not None:
            g1.x = g1.x + torch.randn_like(g1.x) * 0.01
            g2.x = g2.x + torch.randn_like(g2.x) * 0.01
        view_1.append(g1)
        view_2.append(g2)
    return GeomBatch.from_data_list(view_1).to(device), GeomBatch.from_data_list(view_2).to(device)


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

    def train_epoch(self, graphs, batch_size: int, on_batch_end=None) -> float:
        self.encoder.train()
        total_loss = 0.0
        n = len(graphs)
        perm = torch.randperm(n)
        total_batches = max((n + batch_size - 1) // batch_size, 1)
        for step, start in enumerate(range(0, n, batch_size), start=1):
            batch_graphs = [graphs[idx] for idx in perm[start:start + batch_size].tolist()]
            view_1, view_2 = _graph_views(batch_graphs, self.device)
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
) -> None:
    encoder = GraphEncoder(hidden_dim=hidden_dim, graph_backbone=graph_backbone, num_layers=num_layers, dropout=dropout, node_encoding=node_encoding)
    trainer = GraphPretrainer(encoder=encoder, hidden_dim=hidden_dim, proj_dim=128, device=device)
    for epoch in range(1, epochs + 1):
        start = time.time()
        loss = trainer.train_epoch(graphs, batch_size=batch_size, on_batch_end=lambda current, total: _print_progress(f"Graph epoch {epoch}/{epochs}", current, total, start))
        _print_progress(f"Graph epoch {epoch}/{epochs}", 1, 1, start)
        print(f" | loss={loss:.4f}")
    torch.save({"stage": "graph", "graph_encoder": encoder.state_dict()}, out)
    print(f"Saved graph pretrain to {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Contrastive pretraining for the graph encoder")
    parser.add_argument("--data-path", type=str, required=True, help="Graph dataset path (see data_pipeline.data.load_graph_dataset)")
    parser.add_argument("--out", type=str, default="graph_pretrain.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--graph-backbone", type=str, default="gatv2", choices=["gcn", "gat", "gatv2", "gin"])
    parser.add_argument("--node-encoding", type=str, default="dense", choices=["categorical", "dense"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graphs = load_graph_dataset(args.data_path)
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
    )


if __name__ == "__main__":
    main()
