# Hybrid MoLA + MolPROP

This folder contains a standalone hybrid molecular model that:

- uses a graph encoder with selectable backbone: `gcn`, `gat`, `gatv2`, or `gin`
- uses a language branch driven by dense embeddings from `molformer` or `chemberta`
- can disable the language branch entirely with `--no-use-language`
- supports three fusion modes:
  - `mola`: layerwise cross-attention fusion inspired by MoLA
  - `concat`: direct concatenation of graph and language features
  - `molprop`: gated fusion inspired by MolPROP

## Expected input

The trainer expects a saved list of PyG `Data` objects or an `InMemoryDataset` serialized with `torch.save`.

Each graph should provide:

- `x`: node features
- `edge_index`: graph connectivity
- `y`: target property
- `lang` optional: one dense embedding vector per graph

If `x` contains categorical node indices like MolPROP, keep `--node-encoding categorical` and use the correct `--node-vocab-sizes`.

## Training example

```bash
python train.py \
  --data-path data/train_graphs.pt \
  --lang-path data/molformer_embeddings.pt \
  --task regression \
  --graph-backbone gatv2 \
  --language-backbone molformer \
  --fusion mola \
  --use-language \
  --hidden-dim 256 \
  --num-layers 3 \
  --batch-size 32 \
  --epochs 100
```

## Notes

- If you want a graph-only baseline, use `--no-use-language` and `--language-backbone none`.
- If your graph tensors are dense float features instead of categorical indices, switch to `--node-encoding dense`.
- The default categorical node vocabulary sizes match the simplified MolPROP atom representation: atom type + chirality.
