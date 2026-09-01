from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np  # type: ignore[import-not-found]


CLASSIFICATION_LOADERS: Dict[str, str] = {
    "bace_classification": "dc.molnet.load_bace_classification",
    "bbbp": "dc.molnet.load_bbbp",
    "clintox": "dc.molnet.load_clintox",
    "muv": "dc.molnet.load_muv",
    "sider": "dc.molnet.load_sider",
    "tox21": "dc.molnet.load_tox21",
}

REGRESSION_LOADERS: Dict[str, str] = {
    "delaney": "dc.molnet.load_delaney",
    "freesolv": "dc.molnet.load_freesolv",
    "lipo": "dc.molnet.load_lipo",
    "qm7": "dc.molnet.load_qm7",
    "qm8": "dc.molnet.load_qm8",
    "qm9": "dc.molnet.load_qm9",
}

ALL_DATASETS = list(CLASSIFICATION_LOADERS.keys()) + list(REGRESSION_LOADERS.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download DeepChem MoleculeNet datasets into this project's data folder"
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Datasets to download. Use 'all' or any subset of: " + ", ".join(ALL_DATASETS),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "datasets"),
        help="Base folder where raw/ and featurized/ will be created",
    )
    parser.add_argument(
        "--featurizer",
        type=str,
        default="MolGraphConvFeaturizer",
        help="DeepChem featurizer name used for caching the graph dataset",
    )
    parser.add_argument(
        "--splitter",
        type=str,
        default="random",
        help="DeepChem splitter to use when generating the cached splits",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Seed passed to DeepChem splitters",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete any cached raw/ and featurized/ data before downloading. Use this if an exported CSV "
        "looks wrong (e.g. target values look z-scored instead of raw) -- a stale on-disk cache from a "
        "previous run is the usual cause, since the transformers returned alongside a *reloaded* cached "
        "dataset don't reliably match what was actually written to disk.",
    )
    return parser.parse_args()


def _load_deepchem_module():
    try:
        import deepchem as dc  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "deepchem is not installed in the active environment. Install it first or run inside the project's environment."
        ) from exc
    return dc


def _resolve_loader(dc, dataset_name: str):
    if dataset_name in CLASSIFICATION_LOADERS:
        return getattr(dc.molnet, CLASSIFICATION_LOADERS[dataset_name].split(".")[-1]), "classification"
    if dataset_name in REGRESSION_LOADERS:
        return getattr(dc.molnet, REGRESSION_LOADERS[dataset_name].split(".")[-1]), "regression"
    raise ValueError(f"Unknown dataset: {dataset_name}")


def _normalize_dataset_names(raw_names: Iterable[str] | None, interactive: bool) -> List[str]:
    if raw_names is None:
        print("Datasets disponibles:")
        print(", ".join(ALL_DATASETS))
        entered = input("Escribe los datasets separados por espacio o 'all': ").strip()
        raw_names = entered.replace(",", " ").split()

    raw_list = list(raw_names)
    if len(raw_list) == 0:
        return []

    names = [name.strip().lower() for name in raw_list if name.strip()]
    if len(names) == 1 and names[0] == "all":
        return ALL_DATASETS

    invalid = [name for name in names if name not in ALL_DATASETS]
    if invalid:
        raise ValueError(f"Unknown dataset names: {', '.join(invalid)}")
    return names


def _download_one(dc, dataset_name: str, base_dir: Path, featurizer_name: str, splitter: str, seed: int, force: bool = False) -> None:
    loader, task_type = _resolve_loader(dc, dataset_name)

    raw_dir = base_dir / "raw"
    featurized_dir = base_dir / "featurized"
    if force:
        import shutil

        # A stale cache under featurized_dir is the classic way this silently goes wrong:
        # `reload=True` happily reloads a previously-cached (already transformer-applied)
        # DiskDataset from an older run of this script, and undo_transforms() on THAT run's
        # freshly-refit transformers doesn't reliably match what was actually cached to disk.
        # Wiping both dirs forces DeepChem to recompute end-to-end, so what we undo-transform
        # below is guaranteed to be the same y that was just produced.
        for d in (raw_dir, featurized_dir):
            if d.exists():
                shutil.rmtree(d)
        print(f"[{dataset_name}] --force: cleared {raw_dir} and {featurized_dir}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    featurized_dir.mkdir(parents=True, exist_ok=True)

    featurizer = getattr(dc.feat, featurizer_name)()

    print(f"[{dataset_name}] downloading/caching as {task_type} dataset")
    _, datasets, transformers = loader(
        featurizer=featurizer,
        splitter=splitter,
        splitter_seed=seed,
        reload=not force,
        data_dir=str(raw_dir),
        save_dir=str(featurized_dir),
    )

    train_dataset, valid_dataset, test_dataset = datasets
    tasks = list(getattr(train_dataset, "tasks", []) or getattr(valid_dataset, "tasks", []) or getattr(test_dataset, "tasks", []))

    # NOTE: we used to pre-filter `transformers` down to "not getattr(t, 'transform_X', False)"
    # before undoing them, on the theory that only X-transformers should be excluded. That filter
    # is the likely reason freesolv's y stayed z-scored even after undo_transforms() ran: if a
    # transformer this DeepChem version returns doesn't expose `transform_X` the way we assumed
    # (or is a wrapper without that attribute at all), our filter silently dropped the very
    # transformer we needed to undo, leaving `y_transformers=[]` and `undo_transforms` a no-op.
    # dc.trans.undo_transforms already only applies transformers whose own `.transform_y` is set,
    # so just hand it the full, unfiltered list -- don't second-guess it here.
    export_dir = base_dir / "csv"
    export_dir.mkdir(parents=True, exist_ok=True)
    _export_split_csv(dc, train_dataset, export_dir / "train.csv", tasks, transformers, label=f"{dataset_name}/train")
    _export_split_csv(dc, valid_dataset, export_dir / "valid.csv", tasks, transformers, label=f"{dataset_name}/valid")
    _export_split_csv(dc, test_dataset, export_dir / "test.csv", tasks, transformers, label=f"{dataset_name}/test")

    print(
        f"[{dataset_name}] done: train={len(train_dataset.X)} valid={len(valid_dataset.X)} test={len(test_dataset.X)}"
    )
    if transformers:
        print(f"[{dataset_name}] transformers cached: {len(transformers)}")


def _format_value(value):
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return ""
    if isinstance(value, (list, tuple, np.ndarray)):
        array = np.asarray(value).reshape(-1)
        return ";".join("" if (isinstance(item, (float, np.floating)) and np.isnan(item)) else str(item) for item in array)
    return str(value)


def _target_stats_line(label: str, targets: "np.ndarray") -> str:
    arr = np.asarray(targets, dtype="float64").reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return f"  [{label}] target stats: (no numeric targets)"
    mean, std, lo, hi = float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())
    # Heuristic tripwire for exactly the bug this fixed: if undo_transforms() silently no-ops
    # (stale cache, API mismatch, ...) a z-scored NormalizationTransformer output looks like
    # this -- centered near 0 with unit-ish std -- while every real MoleculeNet regression
    # target we ship (ESOL, FreeSolv, Lipophilicity, ...) has |mean| or std well outside this
    # band. False positives are possible for a genuinely tiny/centered raw target; false
    # negatives are not a safety issue since this is a print, not a hard failure.
    suspicious = abs(mean) < 0.5 and 0.5 <= std <= 2.0
    flag = "  <-- WARNING: looks standardized (mean~0, std~1) -- rerun with --force" if suspicious else ""
    return f"  [{label}] target stats: n={arr.size} mean={mean:.3f} std={std:.3f} range=[{lo:.3f}, {hi:.3f}]{flag}"


def _export_split_csv(dc, dataset, output_path: Path, tasks: List[str], transformers: List, label: str = "") -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    smiles = list(getattr(dataset, "ids", []))
    targets = np.asarray(getattr(dataset, "y", []))
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)
    if transformers:
        # dc.molnet loaders return y already transformed (for regression tasks, typically
        # z-scored by a NormalizationTransformer) so the loss is well-scaled during DeepChem's
        # own training. That scale is meaningless outside DeepChem and NOT what papers report
        # RMSE/MAE in -- undo it here so the exported CSV holds the original target units
        # (e.g. raw log-solubility), comparable to SOTA numbers.
        targets = dc.trans.undo_transforms(targets, transformers)

    if len(smiles) != len(targets):
        raise ValueError(
            f"Cannot export CSV: mismatched lengths for {output_path.name} (smiles={len(smiles)}, targets={len(targets)})"
        )

    headers = ["smiles"] + (tasks if tasks else [f"task_{idx}" for idx in range(targets.shape[1])])
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for smi, row in zip(smiles, targets):
            writer.writerow([_format_value(smi)] + [_format_value(value) for value in np.asarray(row).reshape(-1)])

    print(_target_stats_line(label or output_path.stem, targets))


def main() -> None:
    args = parse_args()
    dc = _load_deepchem_module()

    dataset_names = _normalize_dataset_names(args.datasets, interactive=True)
    if not dataset_names:
        print("No datasets selected. Nothing to download.")
        return

    base_dir = Path(args.output_dir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {base_dir}")

    for dataset_name in dataset_names:
        _download_one(
            dc=dc,
            dataset_name=dataset_name,
            base_dir=base_dir / dataset_name,
            featurizer_name=args.featurizer,
            splitter=args.splitter,
            seed=args.seed,
            force=args.force,
        )


if __name__ == "__main__":
    main()
