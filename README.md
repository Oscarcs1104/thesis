# Hybrid MoLA + MolPROP

This folder contains a standalone hybrid molecular model that:

- uses a graph encoder with selectable backbone: `gcn`, `gat`, `gatv2`, or `gin`
- uses a language branch driven by dense embeddings from `molformer` or `chemberta`
- can disable the language branch entirely with `--no-use-language`
- supports three fusion modes:
  - `mola`: layerwise cross-attention fusion inspired by MoLA
  - `concat`: direct concatenation of graph and language features
  - `molprop`: gated fusion inspired by MolPROP

## Folder layout

```
model/            MultimodalModel, GraphEncoder/LanguageEncoder, SmilesDecoder (SELFIES)
training/         train.py (predictor / joint / decoder modes), pretrain_graph.py, pretrain_selfies.py
data_pipeline/    dataset loading, SMILES->PyG conversion, MoleculeNet download
tools/            demo_generate_property.py, ploting.py, test_data_loader.py
data/             raw + cached datasets (unchanged by scripts above)
checkpoints/      saved training runs
```

All entrypoint scripts add the project root to `sys.path`, so they can be run directly, e.g. `python training/train.py ...` from the `test/` folder.

## Expected input

The trainer expects a saved list of PyG `Data` objects or an `InMemoryDataset` serialized with `torch.save`.

Each graph should provide:

- `x`: node features
- `edge_index`: graph connectivity
- `y`: target property
- `smiles` optional: SMILES string, required for the SELFIES decoder (`--use-decoder`)
- `lang` optional: one dense embedding vector per graph (only used when `--language-backbone` is not `chemberta`)

If `x` contains categorical node indices like MolPROP, keep `--node-encoding categorical` and use the correct `--node-vocab-sizes`.

## Training example

Property prediction only (graph encoder + SELFIES/ChemBERTa encoder -> concat -> MLP):

```bash
python training/train.py \
  --data-path data/esol.csv,data/freesolv.csv,data/lipo.csv \
  --task regression \
  --graph-backbone gatv2 \
  --language-backbone chemberta \
  --fusion concat \
  --use-language \
  --hidden-dim 256 \
  --num-layers 3 \
  --batch-size 32 \
  --epochs 100 \
  --training-mode predictor
```

Joint training of the predictor and the SELFIES decoder together:

```bash
python training/train.py --data-path data/esol.csv --use-decoder --training-mode joint
```

Autoregressive decoder-only training on top of an existing predictor checkpoint (freezes everything except the decoder):

```bash
python training/train.py --data-path data/esol.csv --use-decoder --training-mode decoder \
  --load-checkpoint checkpoints/best.pt
```

## Pretraining

```bash
python training/pretrain_graph.py --data-path data/esol.csv --out graph_pretrain.pt
python training/pretrain_selfies.py --smiles-file data/some_smiles.txt --out selfies_pretrain.pt
```

`pretrain_graph.py` output can be fed back into `train.py` via `--graph-pretrained-checkpoint`.
`pretrain_selfies.py` is exploratory/independent — it is not auto-loaded by `train.py`'s ChemBERTa language branch.

## Notes

- If you want a graph-only baseline, use `--no-use-language` and `--language-backbone none`.
- If your graph tensors are dense float features instead of categorical indices, switch to `--node-encoding dense`.
- The default categorical node vocabulary sizes match the simplified MolPROP atom representation: atom type + chirality.
- Checkpoints saved by `train.py` include `args` and `decoder_vocab`, so `tools/demo_generate_property.py` can reload a model without re-specifying every flag.
