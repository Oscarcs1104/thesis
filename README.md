# Graph + SELFIES Multimodal Property Predictor

This folder contains a standalone multimodal molecular model that:

- uses a graph encoder with selectable backbone: `gcn`, `gat`, `gatv2`, or `gin`
- uses a language branch driven by any HuggingFace text encoder (`--language-model-name`), e.g. ChemBERTa or MoLFormer, fed the molecule's raw SMILES
- fuses both branches by concatenation, then predicts the target property with an MLP head
- can disable the language branch entirely with `--no-use-language`
- optionally trains a SELFIES decoder to generate new, valid molecules conditioned on the fused representation and a target property

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
- `smiles` optional: SMILES string. Fed raw (no SELFIES conversion) to the HuggingFace language branch, and required for the SELFIES decoder (`--use-decoder`)

If `x` contains categorical node indices like MolPROP, keep `--node-encoding categorical` and use the correct `--node-vocab-sizes`.

## Training example

Property prediction only (graph encoder + HuggingFace text encoder -> concat -> MLP):

```bash
python training/train.py \
  --data-path data/esol.csv,data/freesolv.csv,data/lipo.csv \
  --task regression \
  --graph-backbone gatv2 \
  --language-backbone huggingface \
  --language-model-name DeepChem/ChemBERTa-77M-MLM \
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
`pretrain_selfies.py` is exploratory/independent — it is not auto-loaded by `train.py`'s HuggingFace language branch.

## Notes

- If you want a graph-only baseline, use `--no-use-language` and `--language-backbone none`.
- Some `trust_remote_code=True` HF repos ship a broken tokenizer `auto_map` (seen with `DeepChem/MoLFormer-c3-1.1B`); `LanguageEncoder` automatically falls back to loading `tokenizer.json` directly in that case.
- If your graph tensors are dense float features instead of categorical indices, switch to `--node-encoding dense`.
- The default categorical node vocabulary sizes match the simplified MolPROP atom representation: atom type + chirality.
- Checkpoints saved by `train.py` include `args` and `decoder_vocab`, so `tools/demo_generate_property.py` can reload a model without re-specifying every flag.
