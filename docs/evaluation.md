# Evaluation protocol

What this project measures, on what data, with which metrics. Keep this in sync
with the code; it is the reference for the thesis' methodology section.

## 1. Datasets

Three MoleculeNet **regression** datasets, downloaded via DeepChem:

| key | DeepChem loader | property | ~N |
|---|---|---|---|
| `esol` | `load_delaney` | log-solubility (mol/L) | 1128 |
| `freesolv` | `load_freesolv` | hydration free energy (kcal/mol) | 642 |
| `lipo` | `load_lipo` | lipophilicity logD | 4200 |

Targets are exported in their **original units** (DeepChem's `undo_transforms` is
applied), so RMSE/MAE are comparable to published numbers.

## 2. Split

**Bemis–Murcko scaffold split**, 80/10/10, from DeepChem
(`--splitter scaffold`). One frozen split per dataset, produced by
`data_pipeline/prepare_all.py` into
`data/deepchem_molnet/<name>/csv/{train,valid,test}.csv` and used *as-is* by
every config (predictor ablations and classical baselines alike). No re-splitting
per seed — so seed variance reflects model init / data order only, not split
luck, and numbers are comparable to MoleculeNet / Chemprop.

Internal `--split {scaffold,random}` in the trainers is a fallback for ad-hoc
single-CSV runs only.

## 3. Predictor — what is compared

`scripts/run_baselines.py` runs one standardized matrix: **3 datasets × 8 trained
configs × 3 seeds** (`2025, 2026, 2027`) + a computed ensemble, identical protocol
per cell (AdamW, warmup→plateau LR, grad-clip 1.0, weight-decay 1e-4, batch 32, up
to 100 epochs, early stop patience 15 on val loss, target standardized on train
only). One CSV, one row per (dataset, config, seed) + a `mean+/-std` summary row.

| config | description |
|---|---|
| `ecfp-xgb` | **reference baseline.** ECFP4 (2048 bit) + ~200 RDKit descriptors (median-imputed, standardized on train) → XGBoost, early stopping on val. |
| `ecfp-mlp` | same features → sklearn MLP (512-256). Neural-on-fingerprints reference. |
| `graph-only` | GIN-E graph encoder alone (OGB atom+bond features). |
| `lang-only-frozen` | ChemBERTa-77M, frozen, mean-pooled → head. |
| `lang-only-unfrozen` | ChemBERTa-77M fine-tuned end to end, lr 1e-5. Unstable on ≤1k molecules — reported as a **negative result**, not a recommended setup. |
| `fusion-concat` | graph ⊕ language, pool-then-concatenate, MLP head (`model/model.py`). |
| `fusion-xattn` | graph ↔ language cross-attention over the full node/token sequences, then pool + head (`model/cross_attention_model.py`). |
| `fusion-moe` | multi-gate mixture-of-experts over the pooled (graph, language) pair (`model/moe_fusion_model.py`). |
| `ensemble` | **the reported final predictor.** A stacked ensemble: a small MLP meta-learner (`baselines/stacking.py`) is trained on the three fusion models' **validation** predictions and evaluated on their **test** predictions. Per seed → RMSE → `mean ± std` over seeds. `ensemble-avg` (plain unweighted mean) is emitted alongside as the reference the stacker has to beat; `ensemble-all` stacks over all 3×3 = 9 checkpoints (one number). |

The three fusion architectures are the comparison of interest — concat vs.
cross-attention vs. gated MoE — and the ensemble of the three is what the thesis
reports as the final model. All eight trained configs go through the *same*
`run_predictor_ablation_training` loop; the standalone
`training/train_cross_attention.py` / `train_moe_fusion.py` entrypoints exist for
ad-hoc single runs.

> **No ZINC pretraining in the matrix.** `training/pretrain_graph.py` and
> `--graph-pretrained-checkpoint` still work if wanted, but a pretrained-encoder
> config is deliberately not part of the reported comparison.

### Metrics (predictor)

- **RMSE** — headline, target units.
- **MAE** — target units.
- **NRMSE** = RMSE / std(train targets). Scale-free, comparable across the three
  datasets. std, not range, so one FreeSolv outlier doesn't distort it.
- reported as mean ± std over the 3 seeds (for `ensemble`, over the 3 per-seed
  ensembled predictions).

### Ensemble method (stacking)

Base models = the three trained fusion predictors, **frozen**. `run_baselines.py`
dumps each one's predictions on **both** the val and test split (in the target's
real units — inverse-standardized). For seed *s*:

1. Meta-features = the 3 base models' scalar predictions (not internal
   representations — with 64–3400 val molecules a 3→8→1 MLP is already near its
   capacity; a 1280-d representation stack would just overfit).
2. Fit the meta-learner (`sklearn` MLP, hidden `(8,)`, strong L2 `alpha=1`,
   `lbfgs`) on `(val base predictions → val targets)`.
3. Predict the test set from the 3 base models' test predictions; `RMSE_s`.

Report `mean ± std` over *s*. **Leakage note:** the meta-learner trains on the
base models' *val* predictions. The base models used val only for early stopping,
not weight updates, so this is a legitimate (mildly optimistic) holdout — much
less leaky than reusing train, where the bases have largely memorised the
targets. A strict out-of-fold scheme is stronger but ~3–5× more base-model
training; it's noted as future work, not the default.

`run_baselines.py` asserts `y_true` matches across the three prediction files
(same frozen split, `shuffle=False`) before stacking. `ensemble-avg` = plain
unweighted mean of the 3 test predictions (reference). `ensemble-all` = one
stacker over all 3×3 = 9 base models.

## 4. Graph features

OGB-style integer feature indices (`data_pipeline/features.py`), embedded by
`model/atom_bond_encoders.py`:

- **atoms** `x` `[N, 9]`: atomic number, chirality, degree, formal charge, #H,
  #radical electrons, hybridization, aromatic, in-ring.
- **bonds** `edge_attr` `[E, 3]`: bond type, stereo, conjugated. Consumed by
  `GINEConv` / `GAT(v2)Conv(edge_dim=…)`; `GCNConv` ignores bonds (documented).

`FEATURE_VERSION` in that file is part of the on-disk graph-cache filename, so a
schema change invalidates stale caches instead of silently reusing them.

## 5. Generation (plan Days 8–10)

A **standalone conditional generator** (`model/conditional_generator.py`),
independent of the predictor:

1. **Pretrain** (`train_generator.py --mode pretrain`): 6-layer decoder-only
   transformer, causal LM over ZINC-250k **SELFIES**, unconditional.
2. **Pseudo-label** ZINC with a trained predictor checkpoint — `fusion-concat`
   (the single cheapest fusion model) is enough here; the ensemble is for the
   reported prediction numbers, not for labelling 250k molecules
   (`data_pipeline/pseudo_label_zinc.py`) → `(smiles, y_pred)`.
3. **Fine-tune** (`--mode finetune`): condition on the **target property value**,
   discretized into 10 quantile bins (edges fit on the dataset's train targets),
   prepended as a learned token. `--cond-dropout 0.1` drops the condition to a
   null bin (classifier-free-guidance style). The conditioning is the *target*,
   never the input molecule — that was the old design's flaw.

### Metrics (generation) — `tools/eval_generation.py`, 10k samples

- **validity** — SELFIES-level (≈100% by construction, sanity check) **and**
  SMILES-level (RDKit parses the decoded SMILES) — the real number.
- **uniqueness** — distinct canonical SMILES / valid.
- **novelty** — fraction of unique-valid not in ZINC-250k, and separately not in
  the dataset train split.
- **internal diversity** — 1 − mean pairwise Tanimoto (ECFP4), capped subset.
- **scaffold diversity** — unique Murcko scaffolds / valid + normalized Shannon
  entropy over scaffolds.
- **FCD** — Fréchet ChemNet Distance vs a ZINC reference sample (needs
  `fcd_torch`; reported `null` if absent).
- **MAD** — mean |predictor(generated) − bin center| over unique-valid molecules,
  per bin and overall, plus the **in-bin rate** (fraction whose predicted
  property lands inside the target bin edges). This is the Day-10 number.

## 6. Reproducing from a clean clone

```bash
pip install -r requirements.txt          # + torch/torch-geometric per that file
pip install deepchem                     # dataset download only
python data_pipeline/download_zinc15.py  # if data/zinc15_250K.csv is absent
python data_pipeline/prepare_all.py      # scaffold splits + graph caches
python scripts/run_baselines.py          # the predictor matrix -> results/baselines.csv
```

Datasets, caches, checkpoints and fresh result CSVs are **not** committed
(`.gitignore`); only `data/zinc15_250K.csv` and
`results/_archive_pre_scaffold/` (superseded numbers, do not cite) are kept.
