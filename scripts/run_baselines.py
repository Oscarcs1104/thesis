"""Standardized predictor matrix.

Same seeds, LR schedule, patience, target standardization and (scaffold) split
across every config and dataset, logged to one CSV -- every row is comparable.

Configs (per dataset):
  ecfp-xgb             ECFP4 + RDKit descriptors -> XGBoost   (classical reference)
  ecfp-mlp             ECFP4 + RDKit descriptors -> sklearn MLP
  graph-only           GIN-E graph encoder alone (OGB features)
  lang-only-frozen     ChemBERTa alone, frozen
  lang-only-unfrozen   ChemBERTa alone, fine-tuned, low lr (unstable on ~500
                       molecules -- kept as a documented negative result)
  fusion-concat        graph (+) language, pool-then-concatenate, MLP head
  fusion-xattn         graph <-> language cross-attention, then pool + head
  fusion-moe           multi-gate mixture-of-experts over the pooled pair
  ensemble             per seed: mean of the three fusion models' test predictions
                       (original units); RMSE then mean +/- std over seeds.
                       Also emits `ensemble-all` = one number over all 9 checkpoints.

Every dataset uses its DeepChem *scaffold* split from data_pipeline/prepare_all.py.

Usage:
    python scripts/run_baselines.py
    python scripts/run_baselines.py --datasets esol --configs ecfp-xgb fusion-concat fusion-xattn fusion-moe ensemble
    python scripts/run_baselines.py --out results/baselines.csv
"""
from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import training.train as train_mod
from baselines.ecfp_baseline import run_ecfp_baseline, train_target_std
from model.cross_attention_model import build_cross_attention_model_from_args
from model.model import build_model_from_args
from model.moe_fusion_model import build_moe_model_from_args
from training.repro import regression_metrics, seed_everything
from training.train import load_predefined_datasets, resolve_predefined_split, run_predictor_ablation_training

DATASETS: Dict[str, str] = {
    "esol": "data/deepchem_molnet/delaney",
    "freesolv": "data/deepchem_molnet/freesolv",
    "lipo": "data/deepchem_molnet/lipo",
}
ECFP_CONFIGS = ["ecfp-xgb", "ecfp-mlp"]
UNIMODAL_CONFIGS = ["graph-only", "lang-only-frozen", "lang-only-unfrozen"]
FUSION_CONFIGS = ["fusion-concat", "fusion-xattn", "fusion-moe"]
ENSEMBLE_CONFIG = "ensemble"
NEURAL_CONFIGS = UNIMODAL_CONFIGS + FUSION_CONFIGS
ALL_CONFIGS = ECFP_CONFIGS + NEURAL_CONFIGS + [ENSEMBLE_CONFIG]
CSV_FIELDS = ["dataset", "config", "seed", "loss", "rmse", "nrmse", "mae", "mse", "elapsed_s"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the standardized predictor matrix")
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--configs", nargs="*", default=ALL_CONFIGS, choices=ALL_CONFIGS)
    parser.add_argument("--seeds", type=int, nargs="*", default=[2025, 2026, 2027])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default="results/baselines.csv")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/baselines")
    parser.add_argument("--append", action="store_true", help="Append to --out instead of overwriting it")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Shared protocol -- identical across every dataset x neural config
# --------------------------------------------------------------------------- #
def _apply_shared_protocol(args: argparse.Namespace, cli: argparse.Namespace) -> None:
    args.epochs = cli.epochs
    args.patience = cli.patience
    args.device = cli.device
    args.lr_schedule = "plateau"
    args.warmup_epochs = 5
    args.plateau_factor = 0.5
    args.plateau_patience = 5
    args.min_lr_ratio = 0.01
    args.grad_clip = 1.0
    args.weight_decay = 1e-4
    args.standardize_target = True
    args.batch_size = 32
    args.deterministic = False
    args.split = "scaffold"  # fallback only -- every dataset here has a predefined scaffold split


def _build_train_args(config: str, cli: argparse.Namespace) -> argparse.Namespace:
    args = train_mod.parse_args([])
    args.hidden_dim = 256
    args.num_layers = 3
    args.dropout = 0.3
    args.graph_backbone = "gin"
    args.language_model_name = "DeepChem/ChemBERTa-77M-MLM"
    _apply_shared_protocol(args, cli)

    if config == "graph-only":
        args.use_graph, args.use_language, args.language_backbone = True, False, "none"
    elif config == "lang-only-frozen":
        args.use_graph, args.use_language, args.freeze_language_backbone = False, True, True
    elif config == "lang-only-unfrozen":
        args.use_graph, args.use_language, args.freeze_language_backbone = False, True, False
        args.lr = 1e-5
    elif config == "fusion-concat":
        args.use_graph, args.use_language = True, True
    elif config == "fusion-xattn":
        args.use_graph, args.use_language = True, True
        args.num_heads = 4
        args.num_cross_layers = 1
    elif config == "fusion-moe":
        args.use_graph, args.use_language = True, True
        args.num_experts = 4
        args.expert_hidden_dim = None
    else:
        raise ValueError(f"Unhandled neural config: {config}")
    return args


def _build_model(config: str, args: argparse.Namespace):
    if config == "fusion-xattn":
        return build_cross_attention_model_from_args(args)
    if config == "fusion-moe":
        return build_moe_model_from_args(args)
    return build_model_from_args(args)  # graph-only / lang-only-* / fusion-concat


def _write_row(writer, handle, row: dict) -> None:
    writer.writerow(row)
    handle.flush()


def _release_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _split_csvs(dataset_dir: str):
    base = Path(dataset_dir) / "csv"
    return str(base / "train.csv"), str(base / "valid.csv"), str(base / "test.csv")


def _preds_path(preds_dir: Path, dataset_name: str, config: str, seed: int) -> Path:
    return preds_dir / f"{dataset_name}_{config}_s{seed}.preds.pt"


def _metric_row(dataset_name, config, seed, m: dict) -> dict:
    return {"dataset": dataset_name, "config": config, "seed": seed,
            **{k: m.get(k) for k in ("loss", "rmse", "nrmse", "mae", "mse", "elapsed_s")}}


def _run_config(dataset_name, dataset_dir, config, cli, datasets, target_std, preds_dir, writer, handle) -> List[Dict[str, float]]:
    per_seed: List[Dict[str, float]] = []

    if config in ECFP_CONFIGS:
        train_csv, val_csv, test_csv = _split_csvs(dataset_dir)
        for p in (train_csv, val_csv, test_csv):
            if not Path(p).exists():
                raise FileNotFoundError(f"missing split file {p} -- run data_pipeline/prepare_all.py")
        kind = "xgb" if config == "ecfp-xgb" else "mlp"
        for seed in cli.seeds:
            start = time.time()
            m = run_ecfp_baseline(train_csv, val_csv, test_csv, kind, seed, target_std)
            m["elapsed_s"] = time.time() - start
            per_seed.append(m)
            _write_row(writer, handle, _metric_row(dataset_name, config, seed, m))
        return per_seed

    train_set, val_set, test_set = datasets
    args = _build_train_args(config, cli)
    criterion = nn.MSELoss()
    for seed in cli.seeds:
        seed_everything(seed)
        args.checkpoint_path = str(Path(cli.checkpoint_dir) / f"{dataset_name}_{config}_s{seed}.pt")
        preds_out = str(_preds_path(preds_dir, dataset_name, config, seed)) if config in FUSION_CONFIGS else None
        model = _build_model(config, args).to(cli.device)
        start = time.time()
        m = run_predictor_ablation_training(model, args, train_set, val_set, test_set, target_std, criterion, predictions_out=preds_out)
        m["elapsed_s"] = time.time() - start
        per_seed.append(m)
        _write_row(writer, handle, _metric_row(dataset_name, config, seed, m))
        del model
        _release_gpu_memory()
    return per_seed


def _load_fusion_preds(preds_dir, dataset_name, seed):
    """(val_true, val_base [Nval,3], test_true, test_base [Ntest,3]) for one seed's 3 fusion models."""
    paths = [_preds_path(preds_dir, dataset_name, c, seed) for c in FUSION_CONFIGS]
    if not all(p.exists() for p in paths):
        missing = [str(p) for p in paths if not p.exists()]
        raise FileNotFoundError(f"missing fusion prediction files: {missing}")
    loaded = [torch.load(p, weights_only=False) for p in paths]
    for split in ("val", "test"):
        ref = loaded[0][split]["y_true"]
        for d in loaded[1:]:
            if not torch.equal(d[split]["y_true"], ref):
                raise RuntimeError(f"ensemble misalignment ({dataset_name}, seed {seed}, {split}): y_true differs across fusion configs")
    val_true = loaded[0]["val"]["y_true"].numpy()
    test_true = loaded[0]["test"]["y_true"].numpy()
    val_base = np.stack([d["val"]["y_pred"].numpy() for d in loaded], axis=1)
    test_base = np.stack([d["test"]["y_pred"].numpy() for d in loaded], axis=1)
    return val_true, val_base, test_true, test_base


def _run_ensemble(dataset_name, cli, target_std, preds_dir, writer, handle) -> Dict[str, List[Dict[str, float]]]:
    """Stacked ensemble: a small MLP meta-learner (baselines/stacking.py) trained on
    the 3 fusion models' VAL predictions, evaluated on their TEST predictions.
    Also emits `ensemble-avg` (plain mean, reference) and `ensemble-all` (stack over
    all 3x3 checkpoints)."""
    from baselines.stacking import stacked_prediction

    out: Dict[str, List[Dict[str, float]]] = {ENSEMBLE_CONFIG: [], "ensemble-avg": []}
    pooled_val_base, pooled_test_base = [], []
    val_true_ref = test_true_ref = None

    for seed in cli.seeds:
        val_true, val_base, test_true, test_base = _load_fusion_preds(preds_dir, dataset_name, seed)
        val_true_ref, test_true_ref = val_true, test_true
        pooled_val_base.append(val_base)
        pooled_test_base.append(test_base)

        stack_pred = stacked_prediction(val_base, val_true, test_base, seed=seed)
        m_stack = regression_metrics(torch.tensor(stack_pred), torch.tensor(test_true), target_std)
        m_stack["loss"] = m_stack["mse"]
        out[ENSEMBLE_CONFIG].append(m_stack)
        _write_row(writer, handle, _metric_row(dataset_name, ENSEMBLE_CONFIG, seed, m_stack))

        m_avg = regression_metrics(torch.tensor(test_base.mean(axis=1)), torch.tensor(test_true), target_std)
        m_avg["loss"] = m_avg["mse"]
        out["ensemble-avg"].append(m_avg)
        _write_row(writer, handle, _metric_row(dataset_name, "ensemble-avg", seed, m_avg))
        print(f"  seed {seed}: stack RMSE={m_stack['rmse']:.4f} | avg RMSE={m_avg['rmse']:.4f}")

    if len(pooled_test_base) == len(cli.seeds) and cli.seeds:
        val_all = np.concatenate(pooled_val_base, axis=1)    # [Nval, 3*nseeds]
        test_all = np.concatenate(pooled_test_base, axis=1)  # [Ntest, 3*nseeds]
        grand_pred = stacked_prediction(val_all, val_true_ref, test_all, seed=0)
        grand = regression_metrics(torch.tensor(grand_pred), torch.tensor(test_true_ref), target_std)
        grand["loss"] = grand["mse"]
        out["ensemble-all"] = [grand]
        _write_row(writer, handle, _metric_row(dataset_name, "ensemble-all", "all", grand))
        print(f"  ensemble-all ({val_all.shape[1]} base models): RMSE={grand['rmse']:.4f}")
    return out


def _summary_row(dataset_name: str, config: str, per_seed: List[Dict[str, float]]) -> dict:
    row = {"dataset": dataset_name, "config": config, "seed": f"mean+/-std(n={len(per_seed)})"}
    for key in ("loss", "rmse", "nrmse", "mae", "mse"):
        vals = np.array([m[key] for m in per_seed if key in m and m[key] == m[key]], dtype="float64")
        row[key] = f"{vals.mean():.6f}+/-{vals.std():.6f}" if vals.size else ""
    row["elapsed_s"] = f"{sum(m.get('elapsed_s', 0.0) for m in per_seed):.1f}"
    return row


def main() -> None:
    cli = parse_args()
    Path(cli.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    preds_dir = Path(cli.checkpoint_dir) / "preds"
    out_path = Path(cli.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ensemble needs the three fusion configs -- pull them in even if not asked for
    configs = list(cli.configs)
    if ENSEMBLE_CONFIG in configs:
        for c in FUSION_CONFIGS:
            if c not in configs:
                configs.append(c)
        configs = [c for c in configs if c != ENSEMBLE_CONFIG] + [ENSEMBLE_CONFIG]

    mode = "a" if cli.append and out_path.exists() else "w"
    with out_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if mode == "w":
            writer.writeheader()
            handle.flush()

        results: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
        overall_start = time.time()
        for dataset_name in cli.datasets:
            dataset_dir = DATASETS[dataset_name]
            results[dataset_name] = {}

            train_csv, _, _ = _split_csvs(dataset_dir)
            target_std = train_target_std(train_csv) if Path(train_csv).exists() else 0.0

            datasets_loaded = None
            if any(c in NEURAL_CONFIGS for c in configs):
                try:
                    predefined = resolve_predefined_split(argparse.Namespace(
                        dataset_dir=dataset_dir, train_path=None, val_path=None, test_path=None))
                    datasets_loaded = load_predefined_datasets(*predefined)
                except Exception as exc:
                    print(f"  cannot load {dataset_name} splits ({exc!r}); neural configs for it will be skipped", flush=True)

            for i, config in enumerate(configs, 1):
                print(f"\n===== [{dataset_name} {i}/{len(configs)}] {config} =====", flush=True)
                run_start = time.time()
                try:
                    if config == ENSEMBLE_CONFIG:
                        for name, ps in _run_ensemble(dataset_name, cli, target_std, preds_dir, writer, handle).items():
                            if ps:
                                results[dataset_name][name] = ps
                                _write_row(writer, handle, _summary_row(dataset_name, name, ps))
                    else:
                        per_seed = _run_config(dataset_name, dataset_dir, config, cli, datasets_loaded, target_std, preds_dir, writer, handle)
                        if per_seed:
                            results[dataset_name][config] = per_seed
                            _write_row(writer, handle, _summary_row(dataset_name, config, per_seed))
                except Exception as exc:
                    print(f"  FAILED: {dataset_name}/{config}: {exc!r}", flush=True)
                    _write_row(writer, handle, {"dataset": dataset_name, "config": config, "seed": "ERROR", "loss": str(exc)[:200]})
                    continue
                print(f"  done in {time.time() - run_start:.1f}s (total {time.time() - overall_start:.1f}s)", flush=True)

    print(f"\nWrote {out_path}")
    print("\n=== RMSE mean (test), by dataset x config ===")
    print(f"{'config':<20s}" + "".join(f"{d:>18s}" for d in cli.datasets))
    display = list(configs)
    if ENSEMBLE_CONFIG in display:
        pos = display.index(ENSEMBLE_CONFIG) + 1
        display[pos:pos] = ["ensemble-avg", "ensemble-all"]
    for config in display:
        cells = []
        for dataset_name in cli.datasets:
            per_seed = results.get(dataset_name, {}).get(config)
            if not per_seed:
                cells.append(f"{'--':>18s}")
                continue
            rmses = np.array([m["rmse"] for m in per_seed], dtype="float64")
            cells.append(f"{rmses.mean():>10.4f}+/-{rmses.std():<6.3f}")
        print(f"{config:<20s}" + "".join(cells))


if __name__ == "__main__":
    main()
