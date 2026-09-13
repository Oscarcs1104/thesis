# Graph + SELFIES Multimodal Property Predictor

This folder contains a standalone multimodal molecular model that:

- uses a graph encoder with selectable backbone: `gcn`, `gat`, `gatv2`, or `gin`
- uses a language branch driven by any HuggingFace text encoder (`--language-model-name`), e.g. ChemBERTa or MoLFormer, fed the molecule's raw SMILES
- fuses both branches by concatenation, then predicts the target property with an MLP head
- can disable the language branch entirely with `--no-use-language`
- optionally trains a SELFIES decoder to generate new, valid molecules conditioned on the fused representation and a target property

## Folder layout

```
thesis_model/     this project's own model, same taxonomy as crossmodal_model/ below:
  model/            MultimodalModel (main concat fusion), GraphEncoder/LanguageEncoder, SmilesDecoder (SELFIES),
                     plus fusion-ablation models: cross_attention_model.py, moe_fusion_model.py, precomputed_molformer_model.py
  train/            train.py (predictor / joint / decoder modes), pretrain_graph.py,
                     plus fusion-ablation trainers: train_cross_attention.py, train_moe_fusion.py, train_precomputed_molformer.py
  generation/       pretrain_selfies.py, demo_generate_property.py, inspect_latent_space.py
  benchmark/        run_baselines.py, native_split.py
crossmodal_model/ the adapted MoLA architecture (Graph+SMILES, A2 SMILES fix, conditional
                   generation) -- same model/train/generation/benchmark layout; see its own README.md
common/           shared utilities used by both models: repro.py (seeding/scheduler/metrics),
                   wandb_utils.py, mol_metrics.py, selfies_vocab.py
data_pipeline/    dataset loading, SMILES->PyG conversion, MoleculeNet/ZINC15 download, MoLFormer embedding precompute
tools/            demo_predict_ablation.py, check_diversity.py, smiles_to_graph.py
data/             raw + cached datasets, splits (official + re-derived), MolFormer embeddings -- shared by both models
checkpoints/      saved training runs (crossmodal_model's under checkpoints/crossmodal/)
```

See [COMMANDS.md](COMMANDS.md) for how to run every script above.

All entrypoint scripts add the project root (`test/`) to `sys.path`, so they can be run directly, e.g. `python thesis_model/train/train.py ...` from the `test/` folder.

## Expected input

The trainer expects a saved list of PyG `Data` objects or an `InMemoryDataset` serialized with `torch.save`.

Each graph should provide:

- `x`: node features
- `edge_index`: graph connectivity
- `y`: target property
- `smiles` optional: SMILES string. Fed raw (no SELFIES conversion) to the HuggingFace language branch, and required for the SELFIES decoder (`--use-decoder`)

If `x` contains categorical node indices like MolPROP, keep `--node-encoding categorical` and use the correct `--node-vocab-sizes`.

## Commands

Every runnable script's usage (data prep, pretraining, training, fusion ablations, generation/evaluation tools) is catalogued in one place: see [COMMANDS.md](COMMANDS.md).

## Notes

- If you want a graph-only baseline, use `--no-use-language` and `--language-backbone none`.
- Some `trust_remote_code=True` HF repos ship a broken tokenizer `auto_map` (seen with `DeepChem/MoLFormer-c3-1.1B`); `LanguageEncoder` automatically falls back to loading `tokenizer.json` directly in that case.
- If your graph tensors are dense float features instead of categorical indices, switch to `--node-encoding dense`.
- The default categorical node vocabulary sizes match the simplified MolPROP atom representation: atom type + chirality.
- Checkpoints saved by `train.py` include `args` and `decoder_vocab`, so `thesis_model/generation/demo_generate_property.py` can reload a model without re-specifying every flag.
