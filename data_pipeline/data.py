from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import csv
import pickle
import os
import json


def _torch_load(path: Path):
    
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _try_load_file(path: Path):
    """Try to load a file with several loaders and return the object or raise the last exception."""
    last_exc = None
    suffix = path.suffix.lower()
    try:
        if suffix in {".pt", ".pth"}:
            return _torch_load(path)
        if suffix in {".npz", ".npy"}:
            return np.load(path, allow_pickle=True)
        if suffix in {".pkl", ".pickle"}:
            with open(path, "rb") as fh:
                return pickle.load(fh)
    except Exception as e:
        last_exc = e

    # Fallback attempts
    try:
        return _torch_load(path)
    except Exception as e:
        last_exc = e
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception as e:
        last_exc = e
    try:
        return np.load(path, allow_pickle=True)
    except Exception as e:
        last_exc = e

    raise last_exc


def _csv_path_to_graphs(csv_path: Path) -> list[Data]:
    """Convert a CSV/CSV.GZ file with a SMILES column into PyG graphs."""
    try:
        from .convert_smiles_to_pyg import smiles_to_data
    except Exception:
        from convert_smiles_to_pyg import smiles_to_data

    open_f = open
    if str(csv_path).lower().endswith(".gz"):
        import gzip

        open_f = gzip.open

    with open_f(csv_path, "rt", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"No header found in CSV file: {csv_path}")

        smiles_col = None
        for name in reader.fieldnames:
            if name and name.lower() in {"smiles", "smile", "canonical_smiles", "smiles_smiles"}:
                smiles_col = name
                break
        if smiles_col is None:
            smiles_col = reader.fieldnames[0]

        target_col = None
        for name in reader.fieldnames:
            if name and name.lower() != smiles_col.lower():
                target_col = name
                break

        data_list: list[Data] = []
        for row in reader:
            smiles = (row.get(smiles_col) or "").strip()
            if not smiles:
                continue
            target = row.get(target_col) if target_col else None
            data = smiles_to_data(smiles, target)
            if data is not None:
                data_list.append(data)

    if not data_list:
        raise ValueError(f"No valid SMILES rows found in {csv_path}")
    return data_list


def convert_deepchem_featurized_to_pyg(root: Path) -> list[Data]:
    """Convert a DeepChem featurized directory (shards with shard-*-X.npy / -y.npy) into a list of PyG Data.

    This is best-effort: tries to extract node features and adjacency from common DeepChem Graph objects.
    If extraction fails for a sample, it will create a placeholder single-node graph with the label.
    """
    import numpy as _np
    import torch as _torch

    root = Path(root)

    # find split dirs that contain shards
    split_dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        files = set(filenames)
        if any(f.startswith("shard-") and f.endswith("-X.npy") for f in files):
            split_dirs.append(Path(dirpath))

    if not split_dirs:
        raise FileNotFoundError(f"No shard files found under {root}")

    # prefer deeper split dir that contains train_dir
    split_dir = sorted(split_dirs, key=lambda p: len(str(p).split(os.sep)), reverse=True)[0]

    # helper to load and concat shards for a split (X or y)
    def _load_shards(prefix: str):
        parts = []
        for f in sorted(split_dir.glob('shard-*-%s.npy' % prefix)):
            parts.append(_np.load(f, allow_pickle=True))
        if not parts:
            return None
        return _np.concatenate(parts, axis=0)

    X = _load_shards('X')
    y = _load_shards('y')

    if X is None or y is None:
        raise FileNotFoundError(f"Could not find X/y shards in {split_dir}")

    data_list: list[Data] = []
    for i in range(len(X)):
        xi = X[i]
        yi = y[i]
        # attempt heuristics to extract node features and adjacency
        node_feats = None
        edge_index = None
        try:
            # dict-like
            if isinstance(xi, dict):
                # common keys
                for k in ('atom_features', 'node_features', 'features', 'atom_features_2', 'atom_feat'):
                    if k in xi:
                        node_feats = _np.asarray(xi[k])
                        break
                for k in ('adjacency', 'adj', 'adj_mat', 'adjacency_matrix'):
                    if k in xi:
                        adj = _np.asarray(xi[k])
                        ii, jj = _np.nonzero(adj)
                        edge_index = _torch.tensor(_np.vstack([ii, jj]), dtype=_torch.long)
                        break
            # tuple/list (node_feats, adj)
            elif isinstance(xi, (list, tuple)) and len(xi) == 2:
                node_feats = _np.asarray(xi[0])
                adj = _np.asarray(xi[1])
                if adj.ndim == 2:
                    ii, jj = _np.nonzero(adj)
                    edge_index = _torch.tensor(_np.vstack([ii, jj]), dtype=_torch.long)
            else:
                # try attributes
                if hasattr(xi, 'atom_features'):
                    node_feats = _np.asarray(getattr(xi, 'atom_features'))
                if hasattr(xi, 'get_atom_features'):
                    node_feats = _np.asarray(xi.get_atom_features())
                if hasattr(xi, 'adjacency'):
                    adj = _np.asarray(getattr(xi, 'adjacency'))
                    ii, jj = _np.nonzero(adj)
                    edge_index = _torch.tensor(_np.vstack([ii, jj]), dtype=_torch.long)
        except Exception:
            node_feats = None
            edge_index = None

        if node_feats is None:
            # fallback: make single-node placeholder
            data = Data()
            data.x = _torch.zeros((1, 1), dtype=_torch.float)
        else:
            data = Data(x=_torch.tensor(_np.asarray(node_feats), dtype=_torch.float))

        # Ensure edge_index is a torch.LongTensor of shape [2, E]. Try more keys if missing.
        if edge_index is None:
            # look for common keys in dict-like xi
            if isinstance(xi, dict):
                for key in ("edge_index", "edges", "bond_index", "edge_list", "edge_idx"):
                    if key in xi:
                        try:
                            arr = _np.asarray(xi[key])
                            if arr.ndim == 2 and arr.shape[0] == 2:
                                edge_index = _torch.tensor(arr, dtype=_torch.long)
                            elif arr.ndim == 2 and arr.shape[1] == 2:
                                edge_index = _torch.tensor(arr.T, dtype=_torch.long)
                        except Exception:
                            edge_index = None
                        if edge_index is not None:
                            break

        if edge_index is None:
            # final fallback: empty edge_index with zero messages
            edge_index = _torch.empty((2, 0), dtype=_torch.long)

        # assign edge_index
        try:
            data.edge_index = edge_index.to(_torch.long)
        except Exception:
            data.edge_index = _torch.tensor(edge_index, dtype=_torch.long)

        # label
        try:
            data.y = _torch.tensor(yi)
        except Exception:
            data.y = _torch.tensor([float(yi)])

        data_list.append(data)

    return data_list


def load_language_features(path: Optional[str]) -> Optional[torch.Tensor]:
    if not path:
        return None

    feature_path = Path(path)
    if not feature_path.exists():
        raise FileNotFoundError(f"Language feature file not found: {path}")

    if feature_path.suffix.lower() == ".npy":
        return torch.from_numpy(np.load(feature_path)).float()

    loaded = _torch_load(feature_path)
    if isinstance(loaded, torch.Tensor):
        return loaded.float()
    if isinstance(loaded, np.ndarray):
        return torch.from_numpy(loaded).float()
    if isinstance(loaded, dict) and "lang" in loaded:
        return torch.as_tensor(loaded["lang"]).float()

    raise ValueError("Unsupported language feature format. Use .pt, .pth or .npy")


def load_graph_dataset(path: str) -> list[Data]:
    raw_path = str(path).strip()
    if not raw_path:
        raise FileNotFoundError("Graph dataset path is empty")

    if "," in raw_path:
        combined: list[Data] = []
        for part in [p.strip() for p in raw_path.split(",") if p.strip()]:
            candidate = Path(part)
            if not candidate.exists():
                candidate = Path(raw_path).parent / part
            if not candidate.exists() and candidate.suffix == "":
                for alt_suffix in (".csv", ".gz"):
                    alt_candidate = candidate.with_suffix(alt_suffix)
                    if alt_candidate.exists():
                        candidate = alt_candidate
                        break
            if not candidate.exists():
                raise FileNotFoundError(f"Graph dataset not found: {part}")
            combined.extend(load_graph_dataset(str(candidate)))
        return combined

    graph_path = Path(raw_path)
    if graph_path.is_file() and graph_path.suffix.lower() in {".csv", ".gz"}:
        cache_path = graph_path.with_suffix(".graphs.pt") if graph_path.suffix.lower() == ".csv" else graph_path.with_name(f"{graph_path.stem}.graphs.pt")
        if cache_path.exists():
            return _torch_load(cache_path)
        graphs = _csv_path_to_graphs(graph_path)
        torch.save(graphs, cache_path)
        print(f"Converted CSV to graph cache: {cache_path}")
        return graphs

    # Prefer user-provided graphs_from_smiles cache if present
    graphs_from_smiles = graph_path / "graphs_from_smiles.pt"
    if graph_path.is_dir() and graphs_from_smiles.exists():
        return _torch_load(graphs_from_smiles)
    # If this looks like a DeepChem featurized folder, attempt conversion to a PyG dataset
    def _has_shard_files(p: Path):
        for root, dirs, files in os.walk(p):
            for f in files:
                if f.startswith("shard-") and f.endswith("-X.npy"):
                    return True
        return False

    # If user passed top-level featurized folder, try to locate and convert
    if graph_path.is_dir() and _has_shard_files(graph_path):
        converted = graph_path / "dataset_converted.pt"
        if converted.exists():
            return _torch_load(converted)
        else:
            data_list = convert_deepchem_featurized_to_pyg(graph_path)
            torch.save(data_list, converted)
            return data_list
    if graph_path.is_dir():
        # Look for common serialized dataset files inside the directory
        patterns = ["**/*.pt", "**/*.pth", "**/*.pkl", "**/*.npz"]
        candidates = []
        for pat in patterns:
            candidates.extend(list(graph_path.glob(pat)))

        # Exclude obvious non-dataset files (transformers, tokenizers, configs, checkpoints)
        exclude_tokens = ("transform", "tokenizer", "chembert", "molformer", "config", "vocab", "weights", "checkpoint", "ckpt", "README", "license", "log")
        candidates = [p for p in candidates if not any(tok in p.name.lower() for tok in exclude_tokens)]

        if not candidates:
            # Fallback: try again without the exclusion filter but avoid obvious transformer files
            all_candidates = []
            for pat in patterns:
                all_candidates.extend(list(graph_path.glob(pat)))
            # remove only exact transformer checkpoint files but keep other candidates
            transform_names = {"transformers.pkl", "transformers.pt", "transformers.pth"}
            fb_candidates = [p for p in all_candidates if p.name.lower() not in transform_names]
            if fb_candidates:
                candidates = fb_candidates
            else:
                # show top-level entries to help debugging
                entries = [p.name for p in graph_path.iterdir()]
                raise FileNotFoundError(
                    f"No dataset file found in directory {path}. Directory contents: {entries}"
                )

        # Sort candidates: prefer names with 'dataset' or 'train' or 'featurized'
        def score(p: Path):
            name = p.name.lower()
            score = 0
            if "dataset" in name:
                score -= 10
            if "train" in name:
                score -= 5
            if "featur" in name:
                score -= 3
            # prefer larger files
            try:
                size = -p.stat().st_size
            except Exception:
                size = 0
            return (score, size)

        candidates = sorted(candidates, key=score)

        # Try to load candidates until one looks like a dataset
        last_excs = {}
        loaded = None
        chosen = None
        for cand in candidates:
            try:
                obj = _try_load_file(cand)
            except Exception as e:
                last_excs[str(cand)] = repr(e)
                continue

            # validate object strictly
            is_valid = False
            # list/tuple of PyG Data
            if isinstance(obj, (list, tuple)) and len(obj) > 0:
                first = obj[0]
                if hasattr(first, "x") and hasattr(first, "edge_index"):
                    is_valid = True
            # torch_geometric InMemoryDataset
            elif hasattr(obj, "data") and hasattr(obj, "slices"):
                is_valid = True
            # DeepChem dataset-like (has X and y)
            elif hasattr(obj, "X") and hasattr(obj, "y"):
                is_valid = True

            if is_valid:
                loaded = obj
                chosen = cand
                print(f"Loading dataset from file: {chosen}")
                break
            else:
                last_excs[str(cand)] = "Loaded object is not a dataset-like structure"

        if loaded is None:
            entries = [p.name for p in graph_path.iterdir()]
            msg = f"No suitable dataset file found in {path}. Tried candidates: {len(candidates)}. Directory contents: {entries}.\nAttempts:\n"
            for k, v in last_excs.items():
                msg += f" - {k}: {v}\n"
            raise FileNotFoundError(msg)
    else:
        if not graph_path.exists():
            raise FileNotFoundError(f"Graph dataset not found: {path}")
        loaded = _torch_load(graph_path)
    if isinstance(loaded, list):
        return loaded
    if isinstance(loaded, tuple):
        return list(loaded)
    if hasattr(loaded, "data") and hasattr(loaded, "slices"):
        return [loaded.get(i) for i in range(len(loaded))]
    if hasattr(loaded, "__len__") and hasattr(loaded, "__getitem__"):
        return [loaded[i] for i in range(len(loaded))]

    raise ValueError("Unsupported graph dataset format. Save a list of PyG Data objects or an InMemoryDataset")


class HybridGraphLangDataset(Dataset):
    def __init__(self, graphs: Sequence[Data], language_features: Optional[torch.Tensor] = None) -> None:
        self.graphs = list(graphs)
        self.language_features = language_features

        if self.language_features is not None and len(self.language_features) != len(self.graphs):
            raise ValueError(
                f"Language features length {len(self.language_features)} does not match graphs length {len(self.graphs)}"
            )

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int) -> Data:
        graph = self.graphs[index]
        # normalize stored graph on access to ensure PyG-compatible types
        graph = _ensure_edge_index_and_types(graph)
        graph = graph.clone()
        if self.language_features is not None:
            graph.lang = self.language_features[index]
        return graph


def _ensure_edge_index_and_types(graph: Data) -> Data:
    import numpy as _np
    import torch as _torch

    # ensure edge_index exists
    e = getattr(graph, 'edge_index', None)
    if e is None:
        graph.edge_index = _torch.empty((2, 0), dtype=_torch.long)
        return graph

    # convert numpy to torch
    if isinstance(e, _np.ndarray):
        e = _torch.from_numpy(e)

    # if it's a list/tuple
    if isinstance(e, (list, tuple)):
        try:
            arr = _np.asarray(e)
            e = _torch.from_numpy(arr)
        except Exception:
            pass

    # ensure torch tensor
    if not isinstance(e, _torch.Tensor):
        try:
            e = _torch.tensor(e)
        except Exception:
            # fallback to empty
            graph.edge_index = _torch.empty((2, 0), dtype=_torch.long)
            return graph

    # ensure dtype long
    if e.dtype != _torch.long:
        try:
            e = e.long()
        except Exception:
            e = e.to(_torch.long)

    # normalize shape: prefer [2, E]
    if e.dim() == 2 and e.size(0) == 2:
        graph.edge_index = e
        return graph
    if e.dim() == 2 and e.size(1) == 2:
        graph.edge_index = e.t().contiguous()
        return graph

    # handle dense adjacency matrices
    try:
        arr = e.numpy()
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            ii, jj = _np.nonzero(arr)
            edge_index = _torch.tensor(_np.vstack([ii, jj]), dtype=_torch.long)
            graph.edge_index = edge_index
            return graph
    except Exception:
        pass

    # final fallback
    graph.edge_index = _torch.empty((2, 0), dtype=_torch.long)
    return graph
