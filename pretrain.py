"""Pretraining utilities: graph contrastive and simple SMILES masked LM.

Usage examples:
  # download ZINC250K
  wget -O datasets/zinc/zinc250k.csv.gz https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/zinc250k.csv.gz

  # convert to graphs (use existing converter)
  python convert_smiles_to_pyg.py --csv datasets/zinc/raw/zinc250k.csv.gz --out datasets/zinc/graphs_from_smiles.pt

  # graph contrastive pretrain
  python pretrain.py --mode graph --data-path datasets/zinc --out checkpoints/zinc_graph_pretrain.pt --epochs 5

  # language MLM pretrain (char-level)
  python pretrain.py --mode lang --smiles-file datasets/zinc/raw/zinc250k.csv.gz --out checkpoints/zinc_lang_pretrain.pt --epochs 3

"""

from pathlib import Path
import argparse
import random
import math
import gzip
import csv
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from typing import List

# Optional W&B handle (set in main)
WAND_B = None


def _load_config_file(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    try:
        import yaml  # type: ignore[import-not-found]

        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        try:
            import json

            with config_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}


def _format_seconds(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _print_progress(prefix: str, current: int, total: int, start_time: float) -> None:
    total = max(total, 1)
    width = 24
    filled = int(width * current / total)
    bar = "=" * filled + "." * (width - filled)
    elapsed = _format_seconds(time.time() - start_time)
    pct = (current / total) * 100.0
    end = "\n" if current >= total else "\r"
    print(f"{prefix} [{bar}] {pct:5.1f}% | elapsed {elapsed}", end=end, flush=True)


def _pretty_component(value: str) -> str:
    value = str(value).strip()
    normalized = value.lower()
    mapping = {
        "molformer": "Molformer",
        "chemberta": "ChemBerta",
        "gatv2": "Gatv2",
        "gat": "Gat",
        "gcn": "Gcn",
        "gin": "Gin",
        "concat": "Concat",
        "mola": "Mola",
        "molprop": "Molprop",
        "none": "None",
    }
    return mapping.get(normalized, value.replace("_", "-").title())


def build_experiment_name(mode: str, language_backbone: str, graph_backbone: str, fusion: str, graph_pretrain_strategy: str = "contrastive") -> str:
    mode = str(mode).lower()
    if mode == "graph":
        strategy_name = _pretty_component(graph_pretrain_strategy)
        return f"Graph-{_pretty_component(graph_backbone)}-{strategy_name}"
    if mode == "lang":
        return f"Lang-{_pretty_component(language_backbone)}"
    return f"{_pretty_component(language_backbone)}-{_pretty_component(graph_backbone)}-{_pretty_component(fusion)}"


def _resolve_checkpoint_path(args, experiment_name: str) -> Path:
    checkpoint_dir = getattr(args, "checkpoint_dir", None)
    out_value = getattr(args, "out", None)

    if checkpoint_dir:
        filename = {
            "lang": "lang_pretrain.pt",
            "graph": "graph_pretrain.pt",
            "both": "pretrain.pt",
        }[args.mode]
        target = Path(checkpoint_dir) / experiment_name / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    if not out_value:
        raise ValueError("Provide either --out or --checkpoint-dir")

    target = Path(out_value)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target

try:
    from data import load_graph_dataset
except Exception:
    # allow running from repo root
    from .data import load_graph_dataset

try:
    from model import GraphEncoder, LanguageEncoder
except Exception:
    from .model import GraphEncoder, LanguageEncoder


def graph_augment(graph, drop_edge_prob=0.1, node_noise_std=0.01):
    g = graph.clone()
    # edge dropout: remove each edge with prob
    ei = g.edge_index
    if ei is None or ei.numel() == 0:
        return g
    mask = torch.rand(ei.size(1)) >= drop_edge_prob
    g.edge_index = ei[:, mask]
    # node feature noise
    if hasattr(g, 'x') and g.x is not None:
        noise = torch.randn_like(g.x) * node_noise_std
        g.x = g.x + noise
    return g


class GraphNCETrainer:
    def __init__(self, encoder: GraphEncoder, hidden_dim: int = 256, proj_dim: int = 128, device='cpu', tau=0.1):
        self.encoder = encoder.to(device)
        self.device = device
        self.tau = tau
        self.proj = nn.Sequential(nn.Linear(hidden_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim)).to(device)
        self.opt = torch.optim.AdamW(list(self.encoder.parameters()) + list(self.proj.parameters()), lr=1e-4)

    def loss_nt_xent(self, z1, z2):
        # z1,z2: [B, D]
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        batch_size = z1.size(0)
        representations = torch.cat([z1, z2], dim=0)  # [2B, D]
        similarity = representations @ representations.t()  # [2B,2B]
        sim = similarity / self.tau
        labels = torch.arange(batch_size, device=self.device)
        targets = torch.cat([labels + batch_size, labels], dim=0)
        # mask self
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=self.device)
        sim.masked_fill_(mask, -9e15)
        loss = F.cross_entropy(sim, targets)
        return loss

    def train_epoch(self, graphs, batch_size: int, on_batch_end=None):
        from torch_geometric.data import Batch as GeomBatch

        self.encoder.train()
        total = 0.0
        n = len(graphs)
        permutation = torch.randperm(n)
        total_batches = max((n + batch_size - 1) // batch_size, 1)
        batch_idx_counter = 0
        for start in range(0, n, batch_size):
            batch_indices = permutation[start:start + batch_size].tolist()
            batch_graphs = [graphs[idx] for idx in batch_indices]

            # create two stochastic graph views from the same batch
            data_list_1 = [graph_augment(graph, drop_edge_prob=0.15, node_noise_std=0.01) for graph in batch_graphs]
            data_list_2 = [graph_augment(graph, drop_edge_prob=0.15, node_noise_std=0.01) for graph in batch_graphs]
            view_1 = GeomBatch.from_data_list(data_list_1).to(self.device)
            view_2 = GeomBatch.from_data_list(data_list_2).to(self.device)

            _, layer_states_1 = self.encoder(view_1.x, view_1.edge_index, view_1.batch)
            _, layer_states_2 = self.encoder(view_2.x, view_2.edge_index, view_2.batch)
            z1 = self.proj(layer_states_1[-1])
            z2 = self.proj(layer_states_2[-1])
            loss = self.loss_nt_xent(z1, z2)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            total += loss.item()
            batch_idx_counter += 1
            if on_batch_end is not None:
                on_batch_end(batch_idx_counter, total_batches)
        return total


class GraphMaskTrainer:
    """Trainer that masks node features and trains the encoder to reconstruct node embeddings."""
    def __init__(self, encoder: GraphEncoder, hidden_dim: int = 256, device='cpu', mask_prob: float = 0.15, lr: float = 1e-4):
        self.encoder = encoder.to(device)
        self.device = device
        self.mask_prob = mask_prob
        self.opt = torch.optim.AdamW(self.encoder.parameters(), lr=lr)

    def train_epoch(self, graphs, batch_size: int, on_batch_end=None):
        from torch_geometric.data import Batch as GeomBatch

        self.encoder.train()
        total = 0.0
        n = len(graphs)
        permutation = torch.randperm(n)
        total_batches = max((n + batch_size - 1) // batch_size, 1)
        batch_idx_counter = 0
        for start in range(0, n, batch_size):
            batch_indices = permutation[start:start + batch_size].tolist()
            batch_graphs = [graphs[idx] for idx in batch_indices]

            # build original and masked versions and a mask tensor over nodes
            orig_list = []
            masked_list = []
            node_masks = []
            for g in batch_graphs:
                orig = g.clone()
                # ensure .x exists
                if getattr(orig, 'x', None) is None:
                    orig.x = torch.zeros((1, 1), dtype=torch.float)
                num_nodes = orig.x.size(0)
                mask = (torch.rand(num_nodes) < self.mask_prob)
                masked = orig.clone()
                # zero-out masked node features (works for dense features)
                try:
                    masked_x = masked.x.clone()
                    masked_x[mask] = 0
                    masked.x = masked_x
                except Exception:
                    masked.x = masked.x
                orig_list.append(orig)
                masked_list.append(masked)
                node_masks.append(mask)

            view_orig = GeomBatch.from_data_list(orig_list).to(self.device)
            view_masked = GeomBatch.from_data_list(masked_list).to(self.device)
            mask_concat = torch.cat([m.to(self.device) for m in node_masks])

            # forward
            node_state_orig, _ = self.encoder(view_orig.x, view_orig.edge_index, view_orig.batch)
            node_state_masked, _ = self.encoder(view_masked.x, view_masked.edge_index, view_masked.batch)

            if mask_concat.any():
                loss = F.mse_loss(node_state_masked[mask_concat], node_state_orig[mask_concat])
            else:
                loss = torch.tensor(0.0, device=self.device)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            total += loss.item()
            batch_idx_counter += 1
            if on_batch_end is not None:
                on_batch_end(batch_idx_counter, total_batches)
        return total


def train_graph(graphs, out: str, graph_backbone: str = 'gatv2', epochs: int = 5, batch_size: int = 64, device='cpu', hidden_dim: int = 256, num_layers: int = 3, dropout: float = 0.3, node_encoding: str = 'dense', node_vocab_sizes=None, config: dict | None = None, strategy: str = 'contrastive'):
    encoder = GraphEncoder(
        hidden_dim=hidden_dim,
        graph_backbone=graph_backbone,
        num_layers=num_layers,
        dropout=dropout,
        node_encoding=node_encoding,
        node_vocab_sizes=node_vocab_sizes,
    )
    if strategy == 'node_mask':
        trainer = GraphMaskTrainer(encoder=encoder, hidden_dim=hidden_dim, device=device)
    else:
        trainer = GraphNCETrainer(encoder=encoder, hidden_dim=hidden_dim, device=device)

    total_batches = max((len(graphs) + batch_size - 1) // batch_size, 1)
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = trainer.train_epoch(
            graphs,
            batch_size=batch_size,
            on_batch_end=lambda current, total: _print_progress(
                f"Graph epoch {epoch+1}/{epochs}", current, total, epoch_start
            ),
        )
        _print_progress(f"Graph epoch {epoch+1}/{epochs}", total_batches, total_batches, epoch_start)
        print(f" | graph_loss={epoch_loss:.4f}")
        global WAND_B
        if WAND_B is not None:
            try:
                WAND_B.log({"epoch": epoch + 1, "graph/epoch": epoch + 1, "graph/loss": epoch_loss}, step=epoch + 1, commit=True)
            except Exception:
                pass

    torch.save({
        "stage": "graph",
        "graph_encoder": encoder.state_dict(),
        "pretrain_config": config or {},
    }, out)
    print("Saved graph pretrain to", out)


class SmilesDataset(Dataset):
    def __init__(self, csv_path):
        self.smiles = []
        open_f = gzip.open if str(csv_path).endswith('.gz') else open
        with open_f(csv_path, 'rt', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            # detect SMILES column
            smiles_col = None
            for c in reader.fieldnames:
                if c.lower() in ('smiles', 'smile', 'smiles_smiles', 'canonical_smiles'):
                    smiles_col = c
                    break
            if smiles_col is None:
                smiles_col = reader.fieldnames[0]
            for row in reader:
                s = row.get(smiles_col)
                if s:
                    self.smiles.append(s)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, i):
        return self.smiles[i]


class CharTokenizer:
    def __init__(self, smiles_list: List[str]):
        chars = set(''.join(smiles_list))
        self.chars = sorted(list(chars))
        self.pad = '<pad>'
        self.mask = '<mask>'
        self.unk = '<unk>'
        self.vocab = [self.pad, self.mask, self.unk] + self.chars
        self.stoi = {s: i for i, s in enumerate(self.vocab)}
        self.itos = {i: s for s, i in self.stoi.items()}

    def encode(self, s: str, max_len: int):
        ids = [self.stoi.get(c, self.stoi[self.unk]) for c in s]
        if len(ids) >= max_len:
            ids = ids[:max_len]
        else:
            ids = ids + [self.stoi[self.pad]] * (max_len - len(ids))
        return ids


class SimpleMaskLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, nlayers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: [B, L]
        x = self.embed(x) * math.sqrt(self.embed.embedding_dim)
        x = x.permute(1, 0, 2)  # seq-first for transformer
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        return self.out(x)


def train_lang(smiles_file: str, out: str, epochs: int = 3, batch_size: int = 128, device='cpu', config: dict | None = None):
    sf = Path(smiles_file)
    print(f"Loading SMILES file: {sf} (exists={sf.exists()})")
    try:
        size = sf.stat().st_size
        print(f"SMILES file size: {size/1024**2:.2f} MB")
    except Exception:
        pass
    ds = SmilesDataset(Path(smiles_file))
    print(f"Loaded {len(ds)} SMILES")
    tokenizer = CharTokenizer(ds.smiles[:5000])
    max_len = 200
    model = SimpleMaskLM(len(tokenizer.vocab)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    # Manual batching to avoid DataLoader sampler using .numpy() internally (which can fail on some Windows envs)
    n = len(ds)
    for epoch in range(epochs):
        total = 0.0
        count = 0
        # create a random permutation of indices using torch (avoid .numpy())
        perm = torch.randperm(n)
        epoch_start = time.time()
        num_batches = (n + batch_size - 1) // batch_size
        for i in range(0, n, batch_size):
            batch_idx = perm[i : i + batch_size].tolist()
            batch_smiles = [ds[idx] for idx in batch_idx]
            enc = [tokenizer.encode(s, max_len) for s in batch_smiles]
            x = torch.tensor(enc, dtype=torch.long, device=device)
            # mask 15%
            mask = (torch.rand(x.shape, device=device) < 0.15) & (x != tokenizer.stoi[tokenizer.pad])
            targets = x.clone()
            x_masked = x.clone()
            x_masked[mask] = tokenizer.stoi[tokenizer.mask]
            logits = model(x_masked)
            loss = F.cross_entropy(logits[mask], targets[mask]) if mask.any() else torch.tensor(0.0, device=device)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            count += 1
            _print_progress(f"Epoch {epoch+1}/{epochs}", count, num_batches, epoch_start)
        avg = total / count if count > 0 else 0.0
        print(f" | loss={avg:.4f} | total_elapsed={_format_seconds(time.time() - epoch_start)}")
        # log to wandb if available
        global WAND_B
        if WAND_B is not None:
            try:
                WAND_B.log({"epoch": epoch + 1, "lang/epoch": epoch + 1, "lang/loss": avg}, step=epoch + 1, commit=True)
            except Exception:
                pass
    torch.save({
        "stage": "lang",
        "model_state_dict": model.state_dict(),
        "pretrain_config": config or {},
    }, out)
    print("Saved language pretrain to", out)


class PairedGraphSmilesDataset(Dataset):
    """Dataset that yields (Data, smiles) pairs from a saved graphs file or parallel lists."""
    def __init__(self, graphs: List):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        smi = getattr(g, 'smiles', None)
        return g, smi


def nt_xent_loss(z1, z2, tau=0.1):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    batch_size = z1.size(0)
    representations = torch.cat([z1, z2], dim=0)
    similarity = representations @ representations.t()
    sim = similarity / tau
    labels = torch.arange(batch_size, device=z1.device)
    targets = torch.cat([labels + batch_size, labels], dim=0)
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)
    sim.masked_fill_(mask, -9e15)
    loss = F.cross_entropy(sim, targets)
    return loss


def train_multimodal(
    graphs,
    out: str,
    epochs: int = 3,
    batch_size: int = 64,
    device: str = 'cpu',
    hidden_dim: int = 256,
    graph_backbone: str = 'gatv2',
    num_layers: int = 3,
    dropout: float = 0.3,
    node_encoding: str = 'dense',
    node_vocab_sizes=None,
    language_backbone: str = 'molformer',
    use_language: bool = True,
    language_model_name: str = 'DeepChem/ChemBERTa-77M-MLM',
    freeze_language_backbone: bool = True,
    contrastive_weight: float = 1.0,
    mlm_weight: float = 1.0,
    proj_dim: int = 128,
    tau: float = 0.1,
    config: dict | None = None,
):
    """Train joint multimodal objective: graph<->SMILES contrastive + optional MLM."""
    from torch_geometric.data import Batch as GeomBatch

    device = torch.device(device)
    max_len = 200
    tokenizer = None
    language_backbone = str(language_backbone).lower()
    use_language = bool(use_language) and language_backbone != 'none'

    if language_backbone == 'chemberta' and use_language:
        lang_model = LanguageEncoder(
            hidden_dim=hidden_dim,
            language_backbone='chemberta',
            num_layers=num_layers,
            dropout=dropout,
            use_language=True,
            language_model_name=language_model_name,
            freeze_language_backbone=freeze_language_backbone,
        ).to(device)
    else:
        smiles_list = [getattr(g, 'smiles', '') or '' for g in graphs]
        tokenizer = CharTokenizer(smiles_list[:5000])
        lang_model = SimpleMaskLM(len(tokenizer.vocab), d_model=hidden_dim, nhead=8, nlayers=max(1, num_layers)).to(device)

    # Graph encoder
    graph_encoder = GraphEncoder(
        hidden_dim=hidden_dim,
        graph_backbone=graph_backbone,
        num_layers=num_layers,
        dropout=dropout,
        node_encoding=node_encoding,
        node_vocab_sizes=node_vocab_sizes,
    )
    graph_encoder = graph_encoder.to(device)

    # projection heads
    graph_proj = nn.Sequential(nn.Linear(hidden_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim)).to(device)
    lang_proj = nn.Sequential(nn.LazyLinear(proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim)).to(device)

    params = list(graph_encoder.parameters()) + list(graph_proj.parameters()) + list(lang_model.parameters()) + list(lang_proj.parameters())
    opt = torch.optim.AdamW(params, lr=1e-4)

    # Manual batching over graphs list to avoid DataLoader sampler .numpy() calls
    n = len(graphs)
    for epoch in range(epochs):
        total_loss = 0.0
        count = 0
        perm = torch.randperm(n)
        epoch_start = time.time()
        num_batches = (n + batch_size - 1) // batch_size
        for i in range(0, n, batch_size):
            batch_idx = perm[i : i + batch_size].tolist()
            batch_graphs = [graphs[idx] for idx in batch_idx]
            # create PyG Batch from list
            gb = GeomBatch.from_data_list(batch_graphs).to(device)
            smiles_batch = [getattr(g, 'smiles', '') or '' for g in batch_graphs]

            if tokenizer is not None:
                enc = [tokenizer.encode(s or '', max_len) for s in smiles_batch]
                x_lang = torch.tensor(enc, dtype=torch.long, device=device)
                # create mask for MLM
                mask = (torch.rand(x_lang.shape, device=device) < 0.15) & (x_lang != tokenizer.stoi[tokenizer.pad])
                targets = x_lang.clone()
                x_masked = x_lang.clone()
                x_masked[mask] = tokenizer.stoi[tokenizer.mask]
            else:
                x_lang = None
                mask = None
                targets = None
                x_masked = None

            # forward graph
            node_state, layer_graph_states = graph_encoder(gb.x, gb.edge_index, gb.batch)
            graph_repr = layer_graph_states[-1]
            z_graph = graph_proj(graph_repr)

            # forward language
            if language_backbone == 'chemberta':
                lang_state, _ = lang_model(smiles_batch, batch_size=len(batch_graphs), device=device)
                z_lang = lang_proj(lang_state)
                mlm_loss = torch.tensor(0.0, device=device)
            else:
                logits = lang_model(x_masked)
                emb = lang_model.embed(x_lang).mean(dim=1)
                z_lang = lang_proj(emb)
                mlm_loss = F.cross_entropy(logits[mask], targets[mask]) if mask.any() else torch.tensor(0.0, device=device)

            # contrastive and mlm
            c_loss = nt_xent_loss(z_graph, z_lang, tau=tau)
            loss = contrastive_weight * c_loss + (mlm_weight * mlm_loss if language_backbone != 'chemberta' else 0.0)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            count += 1
            _print_progress(f"Epoch {epoch+1}/{epochs}", count, num_batches, epoch_start)

        avg = total_loss / count if count > 0 else 0.0
        print(f" | loss={avg:.4f} | total_elapsed={_format_seconds(time.time() - epoch_start)}")
        # wandb logging
        global WAND_B
        if WAND_B is not None:
            try:
                WAND_B.log({"multimodal/epoch": epoch + 1, "multimodal/loss": avg})
            except Exception:
                pass
    # save checkpoint
    torch.save({
        "stage": "both",
        'graph_encoder': graph_encoder.state_dict(),
        'graph_proj': graph_proj.state_dict(),
        'lang_model': lang_model.state_dict(),
        'lang_proj': lang_proj.state_dict(),
        'pretrain_config': config or {},
    }, out)
    print('Saved multimodal checkpoint to', out)
    if WAND_B is not None:
        try:
            WAND_B.save(out)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['graph', 'lang', 'both'], default='graph')
    parser.add_argument('--data-path', type=str, default=None, help='Path to graphs folder (graphs_from_smiles.pt)')
    parser.add_argument('--smiles-file', type=str, default=None, help='CSV of SMILES for language pretraining')
    parser.add_argument('--out', type=str, default=None, help='Explicit checkpoint file path')
    parser.add_argument('--checkpoint-dir', type=str, default=None, help='Directory that will contain per-experiment checkpoints')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--graph-backbone', type=str, default='gatv2', choices=['gcn', 'gat', 'gatv2', 'gin'])
    parser.add_argument('--language-backbone', type=str, default='molformer', choices=['molformer', 'chemberta', 'none'])
    parser.add_argument('--language-model-name', type=str, default='DeepChem/ChemBERTa-77M-MLM')
    parser.add_argument('--freeze-language-backbone', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--fusion', type=str, default='mola', choices=['concat', 'mola', 'molprop'])
    parser.add_argument('--node-encoding', type=str, default='dense', choices=['categorical', 'dense'])
    parser.add_argument('--node-vocab-sizes', type=int, nargs='*', default=None)
    parser.add_argument('--use-language', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--experiment-name', type=str, default=None, help='Optional explicit experiment name')
    parser.add_argument('--contrastive-weight', type=float, default=1.0)
    parser.add_argument('--mlm-weight', type=float, default=0.5)
    parser.add_argument('--proj-dim', type=int, default=128)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--graph-pretrain-strategy', choices=['contrastive', 'node_mask'], default='node_mask', help='Graph pretraining strategy')
    parser.add_argument('--use-wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='pruebas', help='W&B project name')
    parser.add_argument('--wandb-run-name', type=str, default=None, help='W&B run name')

    args = parser.parse_args()

    # Old behavior: use train_config.yaml / train_config.json as the source of defaults.
    base = Path(__file__).resolve().parent
    config_yaml = base / "train_config.yaml"
    config_json = base / "train_config.json"
    raw_cfg = {}
    if config_yaml.exists():
        raw_cfg = _load_config_file(config_yaml)
    elif config_json.exists():
        raw_cfg = _load_config_file(config_json)

    if raw_cfg:
        cfg = vars(args)
        import sys

        def _cli_provided(key: str) -> bool:
            dash = key.replace("_", "-")
            for a in sys.argv[1:]:
                if a == f"--{key}" or a.startswith(f"--{key}=") or a == f"--{dash}" or a.startswith(f"--{dash}="):
                    return True
            return False

        for k, v in raw_cfg.items():
            cfg_key = k.replace("-", "_")
            if not _cli_provided(cfg_key):
                cfg[cfg_key] = v

        from types import SimpleNamespace

        args = SimpleNamespace(**cfg)

    hidden_dim = int(getattr(args, 'hidden_dim', raw_cfg.get('hidden-dim', 256)))
    graph_backbone = str(getattr(args, 'graph_backbone', raw_cfg.get('graph-backbone', 'gatv2')))
    num_layers = int(getattr(args, 'num_layers', raw_cfg.get('num-layers', 3)))
    dropout = float(getattr(args, 'dropout', raw_cfg.get('dropout', 0.3)))
    node_encoding = str(getattr(args, 'node_encoding', raw_cfg.get('node-encoding', 'dense')))
    node_vocab_sizes = getattr(args, 'node_vocab_sizes', raw_cfg.get('node-vocab-sizes', None))
    language_backbone = str(getattr(args, 'language_backbone', raw_cfg.get('language-backbone', 'molformer')))
    use_language = bool(getattr(args, 'use_language', raw_cfg.get('use-language', True)))
    language_model_name = str(getattr(args, 'language_model_name', raw_cfg.get('language-model-name', 'DeepChem/ChemBERTa-77M-MLM')))
    freeze_language_backbone = bool(getattr(args, 'freeze_language_backbone', raw_cfg.get('freeze-language-backbone', True)))
    fusion = str(getattr(args, 'fusion', raw_cfg.get('fusion', 'mola')))
    graph_pretrain_strategy = str(getattr(args, 'graph_pretrain_strategy', 'contrastive'))
    experiment_name = getattr(args, 'experiment_name', None) or build_experiment_name(args.mode, language_backbone, graph_backbone, fusion, graph_pretrain_strategy=graph_pretrain_strategy)
    args.experiment_name = experiment_name

    if not getattr(args, 'out', None) and not getattr(args, 'checkpoint_dir', None):
        args.checkpoint_dir = str(Path(__file__).resolve().parent / 'checkpoints')

    # Print PID and environment info early so user can monitor system (nvidia-smi, task manager)
    import os
    import torch as _torch
    print(f"PID: {os.getpid()}")
    print(f"Python exe: {sys.executable}")
    print(f"torch: {_torch.__version__}, cuda available: {_torch.cuda.is_available()}")

    # determine runtime device and normalize value
    import torch as _torch
    if args.device == 'cuda' and _torch.cuda.is_available():
        # allow user to pick specific GPU later; for now use default GPU 0
        try:
            _torch.cuda.set_device(0)
        except Exception:
            pass
        print(f"Using device: cuda (id=0). torch.cuda.is_available()={_torch.cuda.is_available()}")
    else:
        print(f"Using device: {args.device}. torch.cuda.is_available()={_torch.cuda.is_available()}")

    # fallback to Thesis/KANO/data if not provided
    if args.data_path is None:
        # prefer a data/ folder colocated with this test folder (user workspace)
        local_data = Path(__file__).resolve().parent / "data"
        base_repo = Path(__file__).resolve().parent.parent
        kano_data = base_repo / "KANO" / "data"
        if local_data.exists():
            args.data_path = str(local_data)
            print(f"Pretrain: using local data path: {args.data_path}")
        elif kano_data.exists():
            args.data_path = str(kano_data)
            print(f"Pretrain: using fallback KANO data path: {args.data_path}")

    if args.smiles_file is None:
        # try to find a CSV in the fallback data folder
        if args.data_path:
            p = Path(args.data_path)
            # search for any .csv or .csv.gz in p and immediate subfolders
            found = None
            for ext in ("*.csv.gz", "*.csv"):
                for f in p.rglob(ext):
                    found = f
                    break
                if found:
                    break
            if found:
                args.smiles_file = str(found)
                print(f"Pretrain: autodetected smiles file: {args.smiles_file}")

    # Initialize wandb if requested
    global WAND_B
    if getattr(args, 'use_wandb', False):
        try:
            import wandb

            WAND_B = wandb
            wandb.init(project=args.wandb_project or 'hybrid-mola-molprop', name=args.wandb_run_name or experiment_name, config=vars(args))
            print('Initialized wandb', args.wandb_project, args.wandb_run_name or experiment_name)
        except Exception as e:
            WAND_B = None
            print('wandb init failed or not installed:', e)

    out_target = _resolve_checkpoint_path(args, experiment_name)

    if args.mode in ('lang', 'both'):
        if not args.smiles_file:
            print('smiles-file required for language pretraining')
        else:
            lang_out = out_target if args.mode == 'lang' else out_target.parent / 'lang_pretrain.pt'
            train_lang(args.smiles_file, str(lang_out), epochs=args.epochs, batch_size=args.batch_size, device=args.device, config=vars(args))

    if args.mode in ('graph', 'both'):
        if not args.data_path:
            print('data-path required for graph pretraining')
        else:
            # prepare graphs
            # load or convert graphs; load_graph_dataset already has heuristics, but print status
            print(f"Loading graphs from data path: {args.data_path}")
            data_path_p = Path(args.data_path)
            graphs_pt = data_path_p / "graphs_from_smiles.pt"
            if not graphs_pt.exists():
                # try to find a CSV to convert
                csv_found = None
                for ext in ("*.csv.gz", "*.csv"):
                    matches = list(data_path_p.rglob(ext))
                    if matches:
                        csv_found = matches[0]
                        break
                if csv_found:
                    print(f"No graphs cache found. Converting CSV to PyG graphs: {csv_found} -> {graphs_pt}")
                    import subprocess
                    cmd = [sys.executable, str(Path(__file__).resolve().parent / "convert_smiles_to_pyg.py"), "--csv", str(csv_found), "--out", str(graphs_pt)]
                    try:
                        subprocess.check_call(cmd)
                        print(f"Converted and saved graphs to {graphs_pt}")
                    except subprocess.CalledProcessError as e:
                        print(f"Automatic conversion failed: {e}. Proceeding to loader which may still find other files.")
                else:
                    print(f"No CSV found under {data_path_p}; skipping automatic conversion.")

            graphs = load_graph_dataset(args.data_path)
            print(f"Loaded {len(graphs)} graphs")
            if args.mode == 'graph':
                train_graph(
                    graphs,
                    str(out_target),
                    graph_backbone=graph_backbone,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    device=args.device,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                    node_encoding=node_encoding,
                    node_vocab_sizes=node_vocab_sizes,
                    config=vars(args),
                    strategy=getattr(args, 'graph_pretrain_strategy', 'contrastive'),
                )
            else:
                # mode == 'both' -> run multimodal contrastive+MLM training
                train_multimodal(
                    graphs,
                    str(out_target),
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    device=args.device,
                    hidden_dim=hidden_dim,
                    graph_backbone=graph_backbone,
                    num_layers=num_layers,
                    dropout=dropout,
                    node_encoding=node_encoding,
                    node_vocab_sizes=node_vocab_sizes,
                    language_backbone=language_backbone,
                    use_language=use_language,
                    language_model_name=language_model_name,
                    freeze_language_backbone=freeze_language_backbone,
                    contrastive_weight=args.contrastive_weight,
                    mlm_weight=args.mlm_weight,
                    proj_dim=args.proj_dim,
                    tau=args.tau,
                    config=vars(args),
                )


if __name__ == '__main__':
    main()
