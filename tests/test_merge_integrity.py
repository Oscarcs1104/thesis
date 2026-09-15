"""Post-merge integrity checks (unify: master + eval-redesign).

The merge unified two independently-developed featurizations (8+4 vs 9+3 columns)
and two incompatible ``regression_metrics`` signatures. These tests pin down the
invariants that a careless future edit would break silently rather than loudly.

Run first on any machine before launching training:

    pytest tests/test_merge_integrity.py -v

Needs torch + torch_geometric; the RDKit-dependent test skips if RDKit can't load
(it doesn't on Windows under Smart App Control -- run it on the Linux box).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.features import ATOM_FEATURE_DIMS, BOND_FEATURE_DIMS, FEATURE_VERSION
from thesis_model.model.encoders import GraphEncoder

HIDDEN = 64
N_NODES, N_EDGES, N_GRAPHS = 11, 20, 3


def _fake_graph():
    """A batch of OGB-schema graphs, built without RDKit."""
    x = torch.stack([torch.randint(0, d, (N_NODES,)) for d in ATOM_FEATURE_DIMS], dim=1)
    edge_attr = torch.stack([torch.randint(0, d, (N_EDGES,)) for d in BOND_FEATURE_DIMS], dim=1)
    edge_index = torch.randint(0, N_NODES, (2, N_EDGES))
    batch = torch.arange(N_NODES) % N_GRAPHS
    return x, edge_index, edge_attr, batch


def test_feature_schema_is_ogb():
    """9 atom columns and 3 bond columns -- the OGB standard. master's pre-merge
    8+4 schema is gone; a checkpoint from either schema won't load against the other,
    so this is the number that must never drift silently."""
    assert len(ATOM_FEATURE_DIMS) == 9
    assert len(BOND_FEATURE_DIMS) == 3
    assert FEATURE_VERSION, "FEATURE_VERSION keys the graph cache filename; it can't be empty"


@pytest.mark.parametrize("pool", ["add", "mean", "max", "mean_max"])
def test_graph_encoder_pooling(pool):
    """All four poolings survived the merge (eval-redesign's encoder only had mean;
    HybridMoLA passes pool='add' and would have been silently downgraded)."""
    x, edge_index, edge_attr, batch = _fake_graph()
    enc = GraphEncoder(hidden_dim=HIDDEN, graph_backbone="gin", num_layers=3, pool=pool)
    node_state, layer_states = enc(x, edge_index, batch, edge_attr=edge_attr)
    assert node_state.shape == (N_NODES, HIDDEN)
    assert len(layer_states) == 3
    assert layer_states[0].shape == (N_GRAPHS, HIDDEN)


def test_graph_encoder_signature_is_edge_attr_last():
    """(x, edge_index, batch, edge_attr=None) -- the signature crossmodal_model's
    HybridEncoder calls with. eval-redesign used (x, edge_index, edge_attr, batch);
    mixing them up feeds the batch vector in as bond features."""
    x, edge_index, edge_attr, batch = _fake_graph()
    enc = GraphEncoder(hidden_dim=HIDDEN, graph_backbone="gin", num_layers=2)
    node_a, _ = enc(x, edge_index, batch, edge_attr=edge_attr)
    node_b, _ = enc(x, edge_index, batch)  # edge_attr optional -> zero-filled
    assert node_a.shape == node_b.shape == (N_NODES, HIDDEN)


def test_graph_encoder_absorbs_dead_kwargs():
    """HybridMoLA still forwards node_vocab_sizes / edge_vocab_sizes / node_encoding.
    They're dead after the merge but must not raise."""
    x, edge_index, edge_attr, batch = _fake_graph()
    enc = GraphEncoder(
        hidden_dim=HIDDEN, graph_backbone="gin", num_layers=2, pool="add",
        node_encoding="categorical", node_vocab_sizes=None, edge_vocab_sizes=None,
    )
    node_state, _ = enc(x, edge_index, batch, edge_attr=edge_attr)
    assert node_state.shape == (N_NODES, HIDDEN)


def test_gradients_reach_the_atom_embeddings():
    x, edge_index, edge_attr, batch = _fake_graph()
    enc = GraphEncoder(hidden_dim=HIDDEN, graph_backbone="gin", num_layers=2)
    _, layer_states = enc(x, edge_index, batch, edge_attr=edge_attr)
    layer_states[-1].sum().backward()
    grad = enc.atom_encoder.embeddings[0].weight.grad
    assert grad is not None and float(grad.abs().sum()) > 0


def test_atom_bond_encoders_do_not_mutate_input():
    """The clamp inside the column embeddings must be out-of-place: an in-place
    clamp_ would corrupt the caller's cached graph tensors."""
    from thesis_model.model.atom_bond_encoders import AtomEncoder, BondEncoder

    x, _, edge_attr, _ = _fake_graph()
    x_before, e_before = x.clone(), edge_attr.clone()
    AtomEncoder(HIDDEN)(x)
    BondEncoder(HIDDEN)(edge_attr)
    assert torch.equal(x, x_before)
    assert torch.equal(edge_attr, e_before)


def test_regression_metrics_reports_r2_and_rejects_legacy_range():
    """eval-redesign dropped r2 while master's CSV writers still read it -- that
    would have written empty R2 columns in silence. And the third argument changed
    from (min, max) to std, so the legacy tuple must fail loudly."""
    from common.repro import regression_metrics

    m = regression_metrics(torch.randn(50), torch.randn(50), 1.7)
    assert {"rmse", "mae", "mse", "nrmse", "r2"} <= set(m)

    with pytest.raises(TypeError, match="target_std"):
        regression_metrics(torch.randn(5), torch.randn(5), (0.0, 3.0))


def test_smiles_to_data_matches_the_declared_schema():
    """End-to-end: the featurizer emits exactly the widths the encoders embed."""
    pytest.importorskip("rdkit")
    from data_pipeline.convert_smiles_to_pyg import smiles_to_data

    data = smiles_to_data("Clc1ccccc1", target=2.8)
    assert data is not None
    assert data.x.dtype == torch.long and data.x.size(1) == len(ATOM_FEATURE_DIMS)
    assert data.edge_attr.dtype == torch.long and data.edge_attr.size(1) == len(BOND_FEATURE_DIMS)
    assert int(data.x.max()) < max(ATOM_FEATURE_DIMS)


def test_graph_encoder_consumes_real_molecules():
    pytest.importorskip("rdkit")
    from torch_geometric.loader import DataLoader

    from data_pipeline.convert_smiles_to_pyg import smiles_to_data

    graphs = [smiles_to_data(s, target=0.0) for s in ("Clc1ccccc1", "Fc1ccccc1", "CCO", "C")]
    batch = next(iter(DataLoader([g for g in graphs if g is not None], batch_size=4)))
    enc = GraphEncoder(hidden_dim=HIDDEN, graph_backbone="gin", num_layers=3, pool="add")
    node_state, layer_states = enc(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr)
    assert node_state.size(1) == HIDDEN
    assert layer_states[-1].shape == (4, HIDDEN)  # includes methane: 1 atom, 0 bonds
