# Multimodal molecular property prediction + conditional generation

- **Predictor**: a graph encoder (GIN-E over OGB-style atom/bond features) +
  a HuggingFace text encoder (ChemBERTa on raw SMILES). Three fusion
  architectures are compared — **concat**, **cross-attention**, **gated MoE** —
  and a **stacked ensemble** of the three (MLP meta-learner) is the reported final
  predictor. Unimodal
  ablations (graph-only, language-only frozen / fine-tuned) and a classical
  **ECFP4 + XGBoost / MLP** reference run in the same matrix.
- **Generator** (separate model): a 6-layer decoder-only transformer over SELFIES,
  pretrained on ZINC-250k and fine-tuned to condition on a **target property
  value** (quantile-bin token). Decoupled from the predictor on purpose.

The full evaluation protocol — datasets, scaffold split, every config, every
metric — is in **[docs/evaluation.md](docs/evaluation.md)**.

## Layout

```
thesis_model/     this project's own model, same taxonomy as crossmodal_model/ below:
  model/            MultimodalModel (main concat fusion), GraphEncoder/LanguageEncoder,
                     AtomEncoder/BondEncoder, SmilesDecoder (SELFIES), ConditionalSmilesGenerator,
                     plus fusion-ablation models: cross_attention_model.py, moe_fusion_model.py
  train/            train.py (predictor / joint / decoder modes), pretrain_graph.py,
                     plus fusion-ablation trainers: train_cross_attention.py, train_moe_fusion.py
  generation/       pretrain_selfies.py, train_generator.py, demo_generate_property.py
  benchmark/        run_baselines.py (the standardized matrix)
crossmodal_model/ the adapted MoLA architecture (Graph+SMILES, A2 SMILES fix, conditional
                   generation) -- same model/train/generation/benchmark layout; see its own README.md
common/           shared utilities used by both models: repro.py (seeding/scheduler/metrics),
                   wandb_utils.py, mol_metrics.py, selfies_vocab.py
data_pipeline/    features.py (OGB featurization -- the single source of truth for the graph
                   schema), data.py, convert_smiles_to_pyg.py, prepare_all.py, download_*, splitters.py
baselines/        ecfp_baseline.py (ECFP4 + descriptors -> XGBoost / MLP), stacking.py
tools/            eval_generation.py, demo_predict_ablation.py, check_diversity.py, smiles_to_graph.py
docs/             evaluation.md (the evaluation protocol)
data/             raw + cached datasets, splits (official + re-derived) -- shared by both models
checkpoints/      saved training runs (crossmodal_model's under checkpoints/crossmodal/)
```

See [COMMANDS.md](COMMANDS.md) for how to run every script above.

All entrypoint scripts add the project root to `sys.path`, so they can be run directly,
e.g. `python thesis_model/train/train.py ...` from the repo root.

## Setup

```bash
pip install -r requirements.txt        # read the header: torch + torch-geometric
                                       # install with the right CUDA index first
```

> **Blackwell GPUs (RTX PRO / 50-series):** need PyTorch ≥ 2.7 with a CUDA 12.8+
> build — `pip install torch --index-url https://download.pytorch.org/whl/cu128`.
> Older `cu121`/`cu124` wheels have no `sm_120` kernels.
>
> **Windows:** Smart App Control blocks RDKit's native DLL
> (`ImportError: DLL load failed ... cDataStructs`). Run on WSL2 / Linux / Docker.

## Data (once, on a fresh clone)

```bash
python data_pipeline/prepare_all.py   # downloads raw CSVs + scaffold splits + OGB graph caches
```

Downloads the raw MoleculeNet CSVs from DeepChem's public S3 bucket (no `deepchem`
/ TensorFlow) and writes `data/deepchem_molnet/<name>/csv/{train,valid,test}.csv`
(frozen scaffold split) for `delaney` (=esol), `freesolv`, `lipo`.
`data/zinc15_250K.csv` ships in the repo; `data_pipeline/download_zinc15.py` (the
one script that still needs `deepchem`) only regenerates it.

## Predictor

Standardized matrix -- 3 datasets x 8 trained configs x 3 seeds + the ensemble ->
one CSV (`ecfp-xgb`, `ecfp-mlp`, `graph-only`, `lang-only-frozen`,
`lang-only-unfrozen`, `fusion-concat`, `fusion-xattn`, `fusion-moe`, `ensemble`):

```bash
python thesis_model/benchmark/run_baselines.py     # -> results/baselines.csv
python thesis_model/benchmark/run_baselines.py --datasets esol --configs fusion-concat fusion-xattn fusion-moe ensemble
```

`ensemble` = **stacked**: a small MLP meta-learner (`baselines/stacking.py`) trained
on the three fusion models' *validation* predictions, evaluated on their *test*
predictions. It also reports `ensemble-avg` (plain mean, the reference to beat) and
`ensemble-all` (one stacker over all 3x3 checkpoints). See
[docs/evaluation.md](docs/evaluation.md) for the leakage/holdout details.

Single fusion run:

```bash
python thesis_model/train/train.py   --dataset-dir data/deepchem_molnet/delaney   --graph-backbone gin --epochs 100 --seeds 2025 2026 2027       # fusion-concat
python thesis_model/train/train_cross_attention.py --dataset-dir data/deepchem_molnet/delaney --epochs 100
python thesis_model/train/train_moe_fusion.py      --dataset-dir data/deepchem_molnet/delaney --epochs 100
```

`--no-use-language` -> graph-only; `--no-use-graph` -> language-only.

## Notes

- Graph features are OGB-style integer indices, defined once in
  `data_pipeline/features.py` and consumed by `thesis_model/model/atom_bond_encoders.py`.
  `FEATURE_VERSION` is part of the graph cache filename, so changing the schema
  invalidates stale caches instead of silently reusing them.
- Some `trust_remote_code=True` HF repos ship a broken tokenizer `auto_map` (seen with
  `DeepChem/MoLFormer-c3-1.1B`); `LanguageEncoder` falls back to loading `tokenizer.json`
  directly in that case.
- Checkpoints saved by `train.py` include `args` and `decoder_vocab`, so
  `thesis_model/generation/demo_generate_property.py` can reload a model without
  re-specifying every flag.
