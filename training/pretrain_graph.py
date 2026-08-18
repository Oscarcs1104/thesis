"""Pretraining for the graph encoder (produces a checkpoint usable via
train.py's --graph-pretrained-checkpoint).

Default objective ("mtl"): masked node-attribute prediction + functional-group
prediction + molecular-descriptor prediction, all from the GNN's own node/pooled
states -- inspired by BerMol (C:\\CVAIL\\DTIAM\\code\\BerMol), adapted onto a real
message-passing GraphEncoder instead of a substructure-transformer.

The original contrastive (NT-Xent) objective is still available via
--objective contrastive.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.FilterCatalog import GetFunctionalGroupHierarchy
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch as GeomBatch
from torch_geometric.nn import global_mean_pool

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data_pipeline.data import load_graph_dataset
from model.encoders import GraphEncoder
from training.wandb_utils import add_wandb_args, wandb_finish, wandb_init, wandb_log

RDLogger.DisableLog("rdApp.*")


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


# ---------------------------------------------------------------------------
# Contrastive objective (kept available via --objective contrastive; not the default)
# ---------------------------------------------------------------------------

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


class GraphContrastivePretrainer:
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

    def train_epoch(self, loader: DataLoader, on_batch_end=None) -> Dict[str, float]:
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
        return {"loss": total_loss / max(total_batches, 1)}


# ---------------------------------------------------------------------------
# Multi-task objective (default): masked node attributes + functional groups + descriptors
# ---------------------------------------------------------------------------

_DESCRIPTOR_NAMES = sorted(name for name, _ in Descriptors._descList)


def _mol_descriptors(mol: Chem.Mol) -> np.ndarray:
    calc = MolecularDescriptorCalculator(_DESCRIPTOR_NAMES)
    values = np.array(calc.CalcDescriptors(mol), dtype=np.float64)
    values[~np.isfinite(values)] = 0.0
    return values


def _mol_functional_groups(mol: Chem.Mol) -> List[str]:
    hierarchy = GetFunctionalGroupHierarchy()
    return [match.filterMatch.GetName() for match in hierarchy.GetFilterMatches(mol)]


def _compute_one_label(smiles: str) -> Optional[Tuple[np.ndarray, List[str]]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _mol_descriptors(mol), _mol_functional_groups(mol)


def _build_pretrain_labels(graphs, n_jobs: int = -1):
    """Precompute descriptor + functional-group labels for every graph once
    (RDKit calls are CPU-bound and independent per molecule, so this is
    parallelized across processes -- the same pattern BerMol's vocab.py uses)."""
    indexed = [(i, g.smiles) for i, g in enumerate(graphs) if getattr(g, "smiles", None)]
    print(f"Computing descriptors + functional groups for {len(indexed)} molecules...")
    results = Parallel(n_jobs=n_jobs)(delayed(_compute_one_label)(smi) for _, smi in indexed)

    kept_graphs = []
    raw_descriptors = []
    fg_lists: List[List[str]] = []
    fg_vocab: Dict[str, int] = {}
    for (idx, _), result in zip(indexed, results):
        if result is None:
            continue
        desc, fgs = result
        kept_graphs.append(graphs[idx])
        raw_descriptors.append(desc)
        for fg in fgs:
            if fg not in fg_vocab:
                fg_vocab[fg] = len(fg_vocab)
        fg_lists.append(fgs)

    skipped = len(indexed) - len(kept_graphs)
    if skipped:
        print(f"Skipped {skipped} molecules that failed to parse")

    raw_descriptors_arr = np.stack(raw_descriptors)
    desc_mean = raw_descriptors_arr.mean(axis=0)
    desc_std = raw_descriptors_arr.std(axis=0)
    desc_std[desc_std < 1e-6] = 1.0
    desc_normalized = (raw_descriptors_arr - desc_mean) / desc_std

    fg_labels = np.zeros((len(kept_graphs), len(fg_vocab)), dtype=np.float32)
    for i, fgs in enumerate(fg_lists):
        for fg in fgs:
            fg_labels[i, fg_vocab[fg]] = 1.0

    print(f"Built labels: {len(_DESCRIPTOR_NAMES)} descriptors, {len(fg_vocab)} functional groups")
    return (
        kept_graphs,
        torch.tensor(desc_normalized, dtype=torch.float),
        torch.tensor(fg_labels, dtype=torch.float),
        fg_vocab,
        torch.tensor(desc_mean, dtype=torch.float),
        torch.tensor(desc_std, dtype=torch.float),
    )


class MolPretrainDataset(Dataset):
    def __init__(self, graphs, desc_labels: torch.Tensor, fg_labels: torch.Tensor) -> None:
        self.graphs = graphs
        self.desc_labels = desc_labels
        self.fg_labels = fg_labels

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int):
        return self.graphs[idx], self.desc_labels[idx], self.fg_labels[idx]


def _collate_mtl(items):
    graphs, desc_labels, fg_labels = zip(*items)
    batch = GeomBatch.from_data_list(list(graphs))
    return batch, torch.stack(desc_labels), torch.stack(fg_labels)


def _mask_node_features(x: torch.Tensor, mask_prob: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zero out mask_prob of node rows. Returns (masked_x, mask, original_values_at_mask)."""
    num_nodes = x.size(0)
    mask = torch.rand(num_nodes, device=x.device) < mask_prob
    if not mask.any():
        mask[0] = True
    original = x[mask].clone()
    masked_x = x.clone()
    masked_x[mask] = 0.0
    return masked_x, mask, original


class GraphMultiTaskPretrainer:
    def __init__(
        self,
        encoder: GraphEncoder,
        hidden_dim: int,
        node_feat_dim: int,
        num_fg: int,
        num_desc: int,
        device: str,
        mask_prob: float,
        mask_loss_weight: float,
        fg_loss_weight: float,
        desc_loss_weight: float,
    ) -> None:
        self.encoder = encoder.to(device)
        self.device = torch.device(device)
        self.mask_prob = mask_prob
        self.mask_loss_weight = mask_loss_weight
        self.fg_loss_weight = fg_loss_weight
        self.desc_loss_weight = desc_loss_weight

        self.mask_head = nn.Linear(hidden_dim, node_feat_dim).to(self.device)
        self.fg_head = nn.Linear(hidden_dim, num_fg).to(self.device)
        self.desc_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_desc),
        ).to(self.device)

        params = (
            list(self.encoder.parameters())
            + list(self.mask_head.parameters())
            + list(self.fg_head.parameters())
            + list(self.desc_head.parameters())
        )
        self.opt = torch.optim.AdamW(params, lr=1e-4)
        self.fg_criterion = nn.BCEWithLogitsLoss()
        self.desc_criterion = nn.MSELoss()

    def train_epoch(self, loader: DataLoader, on_batch_end=None) -> Dict[str, float]:
        self.encoder.train()
        self.mask_head.train()
        self.fg_head.train()
        self.desc_head.train()

        totals = {"loss": 0.0, "mask_loss": 0.0, "fg_loss": 0.0, "desc_loss": 0.0}
        total_batches = len(loader)
        for step, (batch, desc_labels, fg_labels) in enumerate(loader, start=1):
            batch = batch.to(self.device)
            desc_labels = desc_labels.to(self.device)
            fg_labels = fg_labels.to(self.device)

            masked_x, mask, original = _mask_node_features(batch.x, self.mask_prob)
            node_state, _ = self.encoder(masked_x, batch.edge_index, batch.batch)
            pooled = global_mean_pool(node_state, batch.batch)

            mask_loss = F.mse_loss(self.mask_head(node_state[mask]), original)
            fg_loss = self.fg_criterion(self.fg_head(pooled), fg_labels)
            desc_loss = self.desc_criterion(self.desc_head(pooled), desc_labels)
            loss = self.mask_loss_weight * mask_loss + self.fg_loss_weight * fg_loss + self.desc_loss_weight * desc_loss

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()

            totals["loss"] += loss.item()
            totals["mask_loss"] += mask_loss.item()
            totals["fg_loss"] += fg_loss.item()
            totals["desc_loss"] += desc_loss.item()
            if on_batch_end is not None:
                on_batch_end(step, total_batches)

        return {k: v / max(total_batches, 1) for k, v in totals.items()}


# ---------------------------------------------------------------------------

def train_graph(
    graphs,
    out: str,
    objective: str = "mtl",
    epochs: int = 5,
    hidden_dim: int = 256,
    num_layers: int = 3,
    dropout: float = 0.3,
    graph_backbone: str = "gatv2",
    device: str = "cpu",
    batch_size: int = 64,
    node_encoding: str = "dense",
    num_workers: int = 0,
    mask_prob: float = 0.15,
    mask_loss_weight: float = 1.0,
    fg_loss_weight: float = 50.0,
    desc_loss_weight: float = 50.0,
    label_jobs: int = -1,
    wandb_run=None,
) -> None:
    encoder = GraphEncoder(hidden_dim=hidden_dim, graph_backbone=graph_backbone, num_layers=num_layers, dropout=dropout, node_encoding=node_encoding)
    extra_checkpoint_data = {}

    if objective == "contrastive":
        trainer = GraphContrastivePretrainer(encoder=encoder, hidden_dim=hidden_dim, proj_dim=128, device=device)
        loader = DataLoader(
            ContrastiveGraphDataset(graphs),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_views,
        )
    else:
        kept_graphs, desc_labels, fg_labels, fg_vocab, desc_mean, desc_std = _build_pretrain_labels(graphs, n_jobs=label_jobs)
        node_feat_dim = kept_graphs[0].x.size(-1)
        trainer = GraphMultiTaskPretrainer(
            encoder=encoder,
            hidden_dim=hidden_dim,
            node_feat_dim=node_feat_dim,
            num_fg=len(fg_vocab),
            num_desc=len(_DESCRIPTOR_NAMES),
            device=device,
            mask_prob=mask_prob,
            mask_loss_weight=mask_loss_weight,
            fg_loss_weight=fg_loss_weight,
            desc_loss_weight=desc_loss_weight,
        )
        loader = DataLoader(
            MolPretrainDataset(kept_graphs, desc_labels, fg_labels),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_mtl,
        )
        extra_checkpoint_data = {
            "fg_vocab": fg_vocab,
            "descriptor_names": _DESCRIPTOR_NAMES,
            "desc_mean": desc_mean,
            "desc_std": desc_std,
        }

    for epoch in range(1, epochs + 1):
        start = time.time()
        metrics = trainer.train_epoch(loader, on_batch_end=lambda current, total: _print_progress(f"Graph epoch {epoch}/{epochs}", current, total, start))
        _print_progress(f"Graph epoch {epoch}/{epochs}", 1, 1, start)
        metrics_str = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f" | {metrics_str}")
        wandb_log(wandb_run, {f"train/{k}": v for k, v in metrics.items()}, step=epoch)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"stage": "graph", "objective": objective, "graph_encoder": encoder.state_dict(), **extra_checkpoint_data}, out)
    print(f"Saved graph pretrain to {out}")
    wandb_finish(wandb_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretraining for the graph encoder")
    parser.add_argument("--data-path", type=str, required=True, help="Graph dataset path (see data_pipeline.data.load_graph_dataset)")
    parser.add_argument("--out", type=str, default="graph_pretrain.pt")
    parser.add_argument(
        "--objective",
        type=str,
        default="mtl",
        choices=["mtl", "contrastive"],
        help="mtl = masked attributes + functional groups + descriptors (default). contrastive = the original NT-Xent objective.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0, help="Parallel CPU processes for batch loading (e.g. 4 if your machine has 4 CPUs)")
    parser.add_argument("--label-jobs", type=int, default=-1, help="Parallel processes for the one-time descriptor/functional-group precompute (mtl only); -1 = all CPUs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--graph-backbone", type=str, default="gatv2", choices=["gcn", "gat", "gatv2", "gin"])
    parser.add_argument("--node-encoding", type=str, default="dense", choices=["categorical", "dense"])
    parser.add_argument("--mask-prob", type=float, default=0.15, help="Fraction of nodes masked for the mtl objective's masked-attribute task")
    parser.add_argument("--mask-loss-weight", type=float, default=1.0)
    parser.add_argument("--fg-loss-weight", type=float, default=50.0)
    parser.add_argument("--desc-loss-weight", type=float, default=50.0)
    add_wandb_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graphs = load_graph_dataset(args.data_path)
    wandb_run = wandb_init(args, config=vars(args))
    train_graph(
        graphs,
        args.out,
        objective=args.objective,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        graph_backbone=args.graph_backbone,
        device=args.device,
        batch_size=args.batch_size,
        node_encoding=args.node_encoding,
        num_workers=args.num_workers,
        mask_prob=args.mask_prob,
        mask_loss_weight=args.mask_loss_weight,
        fg_loss_weight=args.fg_loss_weight,
        desc_loss_weight=args.desc_loss_weight,
        label_jobs=args.label_jobs,
        wandb_run=wandb_run,
    )


if __name__ == "__main__":
    # Re-import as a real module (not __main__) so joblib/loky can pickle the
    # per-molecule worker function by reference instead of by value -- pickling
    # it "by value" out of __main__ drags in an unpicklable RDKit C++ object.
    import training.pretrain_graph as _self

    _self.main()
