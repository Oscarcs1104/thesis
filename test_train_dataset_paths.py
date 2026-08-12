import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train


def test_split_data_paths_handles_comma_separated_values() -> None:
    assert train._split_data_paths("data/esol.csv, data/freesolv.csv ,data/lipo.csv") == [
        "data/esol.csv",
        "data/freesolv.csv",
        "data/lipo.csv",
    ]
