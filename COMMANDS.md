# Comandos

Referencia centralizada de cómo correr cada script del proyecto. Todos se ejecutan desde `test/` (la raíz de este proyecto). Reemplazá los `data/`, `checkpoints/` y SMILES de ejemplo por los tuyos.

## 1. Preparación de datos

### Canonicalizar un CSV (y detectar duplicados)
```
python data_pipeline/canonicalize_csv.py --csv data/esol.csv --out data/esol_canonical.csv
python data_pipeline/canonicalize_csv.py --csv data/esol.csv --out data/esol_canonical.csv --dedupe
```

### Descargar datasets de MoleculeNet (esol, freesolv, lipo, tox21, etc.)
```
python data_pipeline/download_deepchem_datasets.py --datasets esol freesolv lipo
python data_pipeline/download_deepchem_datasets.py --datasets all
```

### Descargar ZINC15 (bucket 1M de DeepChem + submuestra de 500K)
```
python data_pipeline/download_zinc15.py
python data_pipeline/download_zinc15.py --subsample-size 500000 --seed 2025
```

### Convertir un CSV a grafos PyG manualmente (normalmente automático vía caché, rara vez hace falta a mano)
```
python data_pipeline/convert_smiles_to_pyg.py --csv data/mi_dataset.csv --out data/mi_dataset/graphs_from_smiles.pt
```

### Precomputar embeddings congelados de MoLFormer (solo para el ablation MoLA-style)
```
python data_pipeline/precompute_molformer_embeddings.py --data-path data/esol.csv --out data/esol.molformer_emb.pt
```

## 2. Pre-entrenamiento

### Encoder de grafos (GNN)
Objetivo por default `mtl` (atributos enmascarados + grupos funcionales + descriptores RDKit, inspirado en BerMol). `contrastive` (NT-Xent) sigue disponible.
```
python thesis_model/train/pretrain_graph.py --data-path data/zinc15_250K.csv --out checkpoints/graph_pretrain.pt --epochs 50 --device cuda --num-workers 4
python thesis_model/train/pretrain_graph.py --data-path data/zinc15_250K.csv --out checkpoints/graph_pretrain.pt --objective contrastive
```
El checkpoint resultante se carga en `train.py` vía `--graph-pretrained-checkpoint`.

### SELFIES (masked-LM, exploratorio -- no se usa directamente en `train.py`)
```
python thesis_model/generation/pretrain_selfies.py --smiles-file data/some_smiles.txt --out checkpoints/selfies_pretrain.pt
```

## 3. Entrenamiento

### Modelo principal (fusión concat: grafo + encoder de lenguaje HuggingFace)

Solo predicción de propiedad:
```
python thesis_model/train/train.py --data-path data/esol.csv --training-mode predictor --epochs 100 --device cuda
```

Predicción + decoder SELFIES entrenados juntos:
```
python thesis_model/train/train.py --data-path data/esol.csv --use-decoder --training-mode joint --property-context-dropout-prob 0.4
```

Solo el decoder (autoregresivo), congelando todo lo demás, sobre un checkpoint ya entrenado:
```
python thesis_model/train/train.py --data-path data/esol.csv --use-decoder --training-mode decoder --load-checkpoint checkpoints/predictor_esol.pt
```

Con encoder de grafos pre-entrenado (`pretrain_graph.py`) y aumentación de SMILES no-canónico:
```
python thesis_model/train/train.py --data-path data/esol.csv --graph-pretrained-checkpoint checkpoints/graph_pretrain.pt --smiles-augment-prob 0.3
```

Ablation sin la rama de grafos (solo lenguaje):
```
python thesis_model/train/train.py --data-path data/esol.csv --no-use-graph --language-backbone huggingface
```

Con Weights & Biases (aplica a cualquier comando de entrenamiento/pretraining, incluidos los de la sección 4):
```
python thesis_model/train/train.py --data-path data/esol.csv --use-wandb --wandb-project thesis-multimodal --wandb-run-name esol-predictor
```

### Ablations de fusión (predictor-only, sin decoder -- para comparar Test RMSE/NRMSE contra el modelo principal)

Cross-attention entre grafo y lenguaje:
```
python thesis_model/train/train_cross_attention.py --data-path data/esol.csv --epochs 50 --device cuda
```

Mixture-of-Experts (MMoE) sobre la representación fusionada:
```
python thesis_model/train/train_moe_fusion.py --data-path data/esol.csv --epochs 50 --device cuda
```

MoLFormer precomputado y congelado (estilo MoLA -- requiere el paso de precómputo de la sección 1):
```
python data_pipeline/precompute_molformer_embeddings.py --data-path data/esol.csv --out data/esol.molformer_emb.pt
python thesis_model/train/train_precomputed_molformer.py --data-path data/esol.csv --molformer-embeddings-path data/esol.molformer_emb.pt --epochs 50 --device cuda
```

## 4. Generación y evaluación

### Generar moléculas condicionadas en una propiedad + métricas (modelo principal, necesita `--use-decoder`)
```
python thesis_model/generation/demo_generate_property.py --checkpoint-path checkpoints/decoder_esol.pt --smiles "CCO" --num-samples 8
python thesis_model/generation/demo_generate_property.py --checkpoint-path checkpoints/decoder_esol.pt --smiles "CCO" --num-samples 8 --property-values -2.5
python thesis_model/generation/demo_generate_property.py --checkpoint-path checkpoints/decoder_esol.pt --smiles "CCO" --num-samples 8 --reference-data-path data/esol.csv
```

### Predecir con un checkpoint de ablation (cross-attention o MoE, auto-detecta cuál es)
```
python tools/demo_predict_ablation.py --checkpoint-path checkpoints/predictor_esol_cross_attn.pt --smiles "CCO"
```

### Diversidad de un dataset (scaffolds únicos + entropía de Shannon)
```
python tools/check_diversity.py --data-path data/zinc15_250K.csv
```

### Ver el grafo (nodos/aristas) que el modelo arma a partir de un SMILES
```
python tools/smiles_to_graph.py --smiles "CCO"
python tools/smiles_to_graph.py --smiles "CCO" --save-image plots/graph.png
```

### Inspeccionar el espacio latente (fused_feat vs. decoder_latent)
```
python thesis_model/generation/inspect_latent_space.py --checkpoint-path checkpoints/decoder_esol.pt --smiles "CCO"
python thesis_model/generation/inspect_latent_space.py --checkpoint-path checkpoints/decoder_esol.pt --data-path data/esol.csv --max-molecules 300
```
El primero imprime estadísticas del vector usado para predecir (`fused_feat`) y del token de condición que entra al decoder (`decoder_latent`). El segundo embebe una muestra del dataset y guarda dos scatter plots PCA en `plots/` coloreados por la propiedad real, para ver si cada espacio se organiza según la propiedad.

## Notas
- Todos los scripts agregan la raíz del proyecto (`test/`) a `sys.path`, así que corren igual desde cualquier lado siempre que los invoques como `python thesis_model/train/train.py ...` (ruta relativa al script, no `cd` adentro de la carpeta).
- `crossmodal_model/` (la adaptación de MoLA -- benchmark, fix A2 de SMILES, generación condicionada) tiene la misma taxonomía de carpetas (`model/ train/ generation/ benchmark/`) y su propio README con los comandos -- ver `crossmodal_model/README.md`.
