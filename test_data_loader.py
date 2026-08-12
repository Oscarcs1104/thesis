from pathlib import Path

from data import load_graph_dataset


def test_load_graph_dataset_from_comma_separated_csv_paths():
    data_dir = Path(__file__).resolve().parent / "data"
    combined_path = ",".join(
        [str(data_dir / "esol.csv"), str(data_dir / "freesolv.csv"), str(data_dir / "lipo.csv")]
    )

    graphs = load_graph_dataset(combined_path)

    assert len(graphs) == 1128 + 642 + 4200
