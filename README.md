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
model/            MultimodalModel, Graph/LanguageEncoder, AtomEncoder/BondEncoder,
                  SmilesDecoder, ConditionalSmilesGenerator, cross-attn / MoE variants
training/         train.py (predictor), train_cross_attention.py, train_moe_fusion.py,
                  pretrain_graph.py, train_generator.py (pretrain + conditional finetune)
data_pipeline/    features.py (OGB featurization), prepare_all.py, download_*,
                  convert_smiles_to_pyg.py, pseudo_label_zinc.py, splitters.py
baselines/        ecfp_baseline.py (ECFP4 + descriptors -> XGBoost / MLP)
scripts/          run_baselines.py (the standardized matrix)
tools/            eval_generation.py, demo_generate_property.py, check_diversity.py, ploting.py
docs/             evaluation.md
```

Every entrypoint prepends the repo root to `sys.path`, so run them directly
(`python training/train.py ...`) from the repo root.

## Setup

```bash
pip install -r requirements.txt        # read the header: torch + torch-geometric
                                       # install with the right CUDA index first
pip install deepchem                   # dataset download only
```

> **Windows note:** Smart App Control blocks RDKit's native DLL on this machine
> (`ImportError: DLL load failed ... cDataStructs`). Use WSL2 / a Linux box /
> Docker to actually run anything. `torch`, `xgboost`, `torch-geometric` load fine.

## Data (once, on a fresh clone)

```bash
python data_pipeline/download_zinc15.py            # if data/zinc15_250K.csv is missing
python data_pipeline/prepare_all.py               # scaffold splits + OGB graph caches
```

Produces `data/deepchem_molnet/<name>/csv/{train,valid,test}.csv` (frozen scaffold
split) for `delaney` (=esol), `freesolv`, `lipo`. Nothing under `data/` is
committed except `zinc15_250K.csv`.

## Predictor

Standardized matrix — 3 datasets × 8 trained configs × 3 seeds + the ensemble →
one CSV (`ecfp-xgb`, `ecfp-mlp`, `graph-only`, `lang-only-frozen`,
`lang-only-unfrozen`, `fusion-concat`, `fusion-xattn`, `fusion-moe`, `ensemble`):

```bash
python scripts/run_baselines.py                    # -> results/baselines.csv
python scripts/run_baselines.py --datasets esol --configs fusion-concat fusion-xattn fusion-moe ensemble
```

`ensemble` = **stacked**: a small MLP meta-learner (`baselines/stacking.py`) trained
on the three fusion models' *validation* predictions, evaluated on their *test*
predictions. `run_baselines.py` pulls in the three `fusion-*` configs
automatically and also reports `ensemble-avg` (plain mean, the reference to beat)
and `ensemble-all` (one stacker over all 3×3 checkpoints). See
[docs/evaluation.md](docs/evaluation.md) for the leakage/holdout details.

Single fusion run:

```bash
python training/train.py \
  --dataset-dir data/deepchem_molnet/delaney \
  --graph-backbone gin --epochs 100 --seeds 2025 2026 2027       # fusion-concat
python training/train_cross_attention.py --dataset-dir data/deepchem_molnet/delaney --epochs 100
python training/train_moe_fusion.py      --dataset-dir data/deepchem_molnet/delaney --epochs 100
```

`--no-use-language` → graph-only; `--no-use-graph` → language-only.

Optional (not in the reported matrix): graph-encoder pretraining on ZINC via
`training/pretrain_graph.py --objective mtl` + `--graph-pretrained-checkpoint`.

## Generator (plan Days 8–10)

```bash
# 1. unconditional pretrain on ZINC SELFIES
python training/train_generator.py --mode pretrain \
  --smiles-csv data/zinc15_250K.csv --out checkpoints/gen_pretrain.pt

# 2. pseudo-label ZINC with a trained predictor (fusion-concat is enough here)
python data_pipeline/pseudo_label_zinc.py \
  --predictor-checkpoint checkpoints/baselines/esol_fusion-concat_s2025.pt \
  --smiles-csv data/zinc15_250K.csv --out data/zinc15_250K.pseudo_esol.csv

# 3. conditional fine-tune (bins from the dataset's own train targets)
python training/train_generator.py --mode finetune \
  --load-generator checkpoints/gen_pretrain.pt \
  --smiles-csv data/zinc15_250K.pseudo_esol.csv \
  --property-ref-csv data/deepchem_molnet/delaney/csv/train.csv \
  --property-name esol --out checkpoints/gen_esol_finetune.pt

# 4. evaluate: validity / uniqueness / novelty / diversity / FCD / MAD
python tools/eval_generation.py \
  --generator-checkpoint checkpoints/gen_esol_finetune.pt \
  --predictor-checkpoint checkpoints/baselines/esol_fusion-concat_s2025.pt \
  --train-csv data/deepchem_molnet/delaney/csv/train.csv \
  --num-samples 10000 --out results/generation_esol.json
```

## Status

- [x] OGB-standard graph features (GIN-E with bond features)
- [x] Scaffold split enforced everywhere; one results table
- [x] Classical ECFP4 baseline (XGBoost / MLP) as a first-class config
- [x] Three fusion architectures compared (concat / cross-attention / MoE); ensemble = final predictor
- [x] RMSE headline metric; NRMSE = RMSE / train-std
- [x] Data reproducible from a clean clone (`prepare_all.py`); stale results archived
- [x] Conditional generator that actually conditions on the target property
- [ ] run the matrix + generator on GPU / WSL2 and fill `results/`
