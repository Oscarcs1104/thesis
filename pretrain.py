"""Minimal pretraining utilities for the thesis multimodal setup."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch as GeomBatch

try:
    from .data import load_graph_dataset
except Exception:  # pragma: no cover
    from data import load_graph_dataset

try:
    from .encoders import GraphEncoder
except Exception:  # pragma: no cover
    from encoders import GraphEncoder


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


class CharTokenizer:
    def __init__(self, smiles_list) -> None:
        chars = sorted(set("".join(smiles_list)))
        self.pad = "<pad>"
        self.mask = "<mask>"
        self.unk = "<unk>"
        self.vocab = [self.pad, self.mask, self.unk] + chars
        self.stoi = {token: idx for idx, token in enumerate(self.vocab)}

    def encode(self, text: str, max_len: int) -> list[int]:
        ids = [self.stoi.get(char, self.stoi[self.unk]) for char in text]
        if len(ids) >= max_len:
            return ids[:max_len]
        return ids + [self.stoi[self.pad]] * (max_len - len(ids))


class MaskedLanguagePretrainer(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        return self.proj(h)


def train_graph(graphs, out: str, hidden_dim: int = 256, num_layers: int = 3, dropout: float = 0.3, graph_backbone: str = "gatv2", device: str = "cpu", batch_size: int = 64, node_encoding: str = "dense") -> None:
    encoder = GraphEncoder(hidden_dim=hidden_dim, graph_backbone=graph_backbone, num_layers=num_layers, dropout=dropout, node_encoding=node_encoding)
    trainer = GraphPretrainer(encoder=encoder, hidden_dim=hidden_dim, proj_dim=128, device=device)
    for epoch in range(1, 6):
        start = time.time()
        loss = trainer.train_epoch(graphs, batch_size=batch_size, on_batch_end=lambda current, total: _print_progress(f"Graph epoch {epoch}/5", current, total, start))
        _print_progress(f"Graph epoch {epoch}/5", 1, 1, start)
        print(f" | loss={loss:.4f}")
    torch.save({"stage": "graph", "graph_encoder": encoder.state_dict()}, out)
    print(f"Saved graph pretrain to {out}")


def train_lang(smiles_file: str, out: str, epochs: int = 3, batch_size: int = 64, device: str = "cpu") -> None:
    path = Path(smiles_file)
    with path.open("r", encoding="utf-8") as handle:
        texts = [line.strip() for line in handle if line.strip()]
    tokenizer = CharTokenizer(texts[:5000])
    model = MaskedLanguagePretrainer(len(tokenizer.vocab), hidden_dim=256).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    for epoch in range(1, epochs + 1):
        start = time.time()
        total_loss = 0.0
        for batch_idx in range(0, len(texts), batch_size):
            batch = texts[batch_idx:batch_idx + batch_size]
            ids = torch.tensor([tokenizer.encode(text, 128) for text in batch], dtype=torch.long, device=device)
            mask = (torch.rand_like(ids.float()) < 0.15) & (ids != tokenizer.stoi[tokenizer.pad])
            masked_ids = ids.clone()
            masked_ids[mask] = tokenizer.stoi[tokenizer.mask]
            logits = model(masked_ids)
            loss = F.cross_entropy(logits[mask], ids[mask]) if mask.any() else torch.tensor(0.0, device=device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Lang epoch {epoch}/{epochs} | loss={total_loss / max(1, len(texts) // batch_size):.4f}")
    torch.save({"stage": "lang", "model_state_dict": model.state_dict()}, out)
    print(f"Saved language pretrain to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal pretraining entrypoint")
    parser.add_argument("--mode", choices=["graph", "lang", "both"], default="graph")
    parser.add_argument("--data-path", type=str, default=None, help="Folder with graphs_from_smiles.pt")
    parser.add_argument("--smiles-file", type=str, default=None, help="Text file with one SMILES per line")
    parser.add_argument("--out", type=str, default="pretrain.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--graph-backbone", type=str, default="gatv2", choices=["gcn", "gat", "gatv2", "gin"])
    parser.add_argument("--node-encoding", type=str, default="dense", choices=["categorical", "dense"])
    args = parser.parse_args()

    if args.mode in {"graph", "both"}:
        if args.data_path is None:
            raise SystemExit("--data-path is required for graph pretraining")
        graphs = load_graph_dataset(args.data_path)
        train_graph(graphs, args.out, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout, graph_backbone=args.graph_backbone, device=args.device, batch_size=args.batch_size, node_encoding=args.node_encoding)

    if args.mode in {"lang", "both"}:
        if args.smiles_file is None:
            raise SystemExit("--smiles-file is required for language pretraining")
        train_lang(args.smiles_file, args.out if args.mode == "lang" else str(Path(args.out).with_name("lang_pretrain.pt")), epochs=args.epochs, batch_size=args.batch_size, device=args.device)


if __name__ == "__main__":
    main()
