import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def test_load_graph_dataset_from_scaffold_split_csv():
    """Smoke test: load one predefined split CSV into OGB-style PyG graphs.

    Skips on a fresh clone where data_pipeline/prepare_all.py hasn't run yet.
    """
    train_csv = ROOT / "data" / "deepchem_molnet" / "delaney" / "csv" / "train.csv"
    if not train_csv.exists():
        pytest.skip("run data_pipeline/prepare_all.py first (needs deepchem)")

    from data_pipeline.data import load_graph_dataset
    from data_pipeline.features import ATOM_FEATURE_DIMS, BOND_FEATURE_DIMS

    graphs = load_graph_dataset(str(train_csv))
    assert len(graphs) > 100
    g = graphs[0]
    assert g.x.dim() == 2 and g.x.size(1) == len(ATOM_FEATURE_DIMS)
    assert g.edge_attr.size(1) == len(BOND_FEATURE_DIMS)
    assert g.x.dtype.is_floating_point is False  # integer feature indices
    assert isinstance(g.smiles, str) and g.smiles
