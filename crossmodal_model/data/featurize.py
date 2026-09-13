import numpy as np
import torch
from torch_geometric.data import Data


def build_vocab(smiles_list):
    unique_chars = sorted(set("".join(smiles_list)))
    vocab = {char: idx + 1 for idx, char in enumerate(unique_chars)}
    vocab["<pad>"] = 0
    return vocab


def prepare_data(dataset, fp_molformer, smiles, vocab, max_sm_len=100):
    if not (len(dataset.X) == len(fp_molformer) == len(smiles)):
        raise ValueError("Data length mismatch while preparing graph/molecule features.")

    data_list = []
    dataset_weights = getattr(dataset, "w", None)

    for i, (graph, y) in enumerate(zip(dataset.X, dataset.y)):
        x = torch.tensor(graph.node_features, dtype=torch.float32)
        edge_index = torch.tensor(graph.edge_index, dtype=torch.long)
        fp2 = torch.tensor(fp_molformer[i], dtype=torch.float32)

        sm_idx = [vocab.get(char, 0) for char in smiles[i][:max_sm_len]]
        if len(sm_idx) < max_sm_len:
            sm_idx.extend([0] * (max_sm_len - len(sm_idx)))

        y_array = np.asarray(y, dtype=np.float32).reshape(-1)
        y_mask = np.isfinite(y_array).astype(np.float32)
        y_array = np.nan_to_num(y_array, nan=0.0)

        if dataset_weights is None:
            w_array = np.ones_like(y_array, dtype=np.float32)
        else:
            w_array = np.asarray(dataset_weights[i], dtype=np.float32).reshape(-1)
            w_array = np.nan_to_num(w_array, nan=0.0)
        w_array = w_array * y_mask

        sm = torch.tensor(sm_idx, dtype=torch.long)
        y_tensor = torch.tensor(y_array, dtype=torch.float32)
        w_tensor = torch.tensor(w_array, dtype=torch.float32)
        data = Data(
            x=x,
            edge_index=edge_index,
            fp2=fp2.unsqueeze(0),
            sm=sm.unsqueeze(0),
            y=y_tensor,
            w=w_tensor,
        )
        data_list.append(data)

    return data_list
