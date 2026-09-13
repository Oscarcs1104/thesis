# crossmodal_model -- MoLA (referencia), benchmark, fix de SMILES (A2) y generación condicionada

Copia adaptada de `MoLA/model_MoLA.py` + `MoLA/main_Reg_MoLA.py` (repo externo clonado en
`C:\CVAIL\Thesis\MoLA`). Arrancó como una réplica fiel de MoLA para comparar contra
`thesis_model/` (el modelo principal de la tesis); ahora también incluye una rama de
generación molecular condicionada a propiedad, construida sobre el mismo encoder
compartido. Vive junto a `thesis_model/` con la misma taxonomía de carpetas
(`model/ train/ generation/ benchmark/`); lo que ambos comparten (datos, splits, utils de
entrenamiento) vive fuera de los dos, en `data/` y `common/`.

## Layout

- `model/` -- arquitectura: `encoder.py` (Encoder: grafo GINConv + SMILES char-level,
  Graph+SMILES only), `mola.py` (MoLA: Encoder + cross-attention entre capas + cabeza
  final). El branch original de MoLA de tercera modalidad (fp clásico + MolFormer
  congelado, fusionados con un Dendritic Neuron Model) se eliminó por completo -- no se
  usa en esta tesis. `positional_smiles={False,True}`: `False` reproduce el
  `sm_transformer` original de MoLA tal cual (incluido un bug preexistente de eje
  batch/secuencia); `True` es el fix A2 (positional embedding + `batch_first` correcto +
  padding mask + pooling enmascarado) -- **obligatorio** para la rama de generación,
  opcional para regresión.
- `data/featurize.py` -- `build_vocab`/`prepare_data` de MoLA, reusados tal cual por todo
  lo demás. (El `run_experiment` original de MoLA -- que dependía de la tercera
  modalidad -- se eliminó: no lo usaba ningún script propio.)
- `train/core.py` -- featurización del split oficial + lookup de embeddings MolFormer +
  el loop de train/eval (`train_one_epoch`/`evaluate`), compartido por los dos scripts de
  `benchmark/`.
- `benchmark/scaffold_fixed.py` -- benchmark de regresión (Graph+SMILES-only) sobre el
  split oficial FIJO de la tesis (mismas CSVs que `results/fase2_baselines.csv`).
  `--positional-smiles` alterna entre MoLA original y el fix A2. Resultados:
  `results/mola/NewTest_benchmark.csv` (original) /
  `results/mola/NewTest_a2fix_benchmark.csv` (A2).
- `benchmark/native_split.py` -- mismo benchmark pero con la organización de datos
  *nativa* de MoLA (`dc.splits.ScaffoldSplitter`/`RandomSplitter`, re-derivado por seed,
  no el split fijo). Sirvió para chequear qué tan sensible era la ventaja de MoLA al
  split particular usado -- ver `results/mola/NewTest_<scaffold|random>split_benchmark.csv`.
  Vuelve a generar `data/rederived_splits/<splitter>/` (puede borrarse tras usarlo, se
  regenera solo).
- `generation/decoder.py` -- decoder de generación condicionada a propiedad, estilo
  Chemformer: cross-attention sobre el memory sin poolear (nodos de grafo + caracteres
  SMILES + un token de propiedad proyectado), no sobre un vector único. Vocabulario
  SELFIES reusado de `common/selfies_vocab.py` (garantiza validez).
- `generation/train.py` -- entrena esa rama (encoder compartido + decoder, fine-tuning
  completo) sobre un dataset, con evaluación generativa por MUESTREO
  (validez/unicidad/novedad vía `common/mol_metrics.py`). Guarda candidatos en
  `results/mola/mola_generation_<dataset>_samples.csv`.
- `generation/eval_conditioning.py` -- chequeo semántico del conditioning: genera bajo
  −2σ/real/+2σ, predice la propiedad de cada candidato con un predictor INDEPENDIENTE
  (`checkpoints/crossmodal/mola_graph_smiles_a2_smiles_fix/freesolv_..._s2025.pt`, no el
  modelo generativo), reporta correlación solicitado-vs-predicho + % de acierto
  direccional.
- `generation/score_property.py` -- agrega `predicted_property` a los samples ya
  generados por `generation/train.py`, con el mismo predictor independiente.
- `generation/from_smiles.py` -- herramienta ad-hoc: `--smiles X --property-values V1 V2
  ...` → candidatos generados + su propiedad predicha por el mismo predictor
  independiente.

## Estado de los checkpoints (`checkpoints/crossmodal/`)

Solo se conservan los que algún script todavía referencia en código:
- `checkpoints/crossmodal/generation/freesolv_mola_gen_s2025.pt` -- el generador
  entrenado (100 epochs, patience 15), usado por `eval_conditioning.py`/`from_smiles.py`.
- `checkpoints/crossmodal/mola_graph_smiles_a2_smiles_fix/freesolv_..._s2025.pt` -- el
  predictor independiente que usan esos mismos dos scripts.

El resto de los checkpoints de regresión (ESOL/Lipo × 3 seeds, y las otras seeds/variantes
de FreeSolv) se borraron -- sus resultados ya están en los CSV de `results/` y son baratos
de regenerar (`benchmark/scaffold_fixed.py`, unos minutos por dataset/seed) si hace falta
recargarlos.

**Nota post-eliminación de DNM/MolFormer:** `checkpoints/crossmodal/NewTest_molformer/` y
`NewTest_randomsplit_molformer/` (y sus CSV `results/mola/*_molformer*.csv`) corresponden
a la rama de tercera modalidad ya eliminada del código -- esos checkpoints ya no cargan
contra la arquitectura actual (`Encoder`/`MoLA` sin `molformer_mlp`). Se dejaron en disco
como resultado histórico; bórralos a mano si no los necesitás.

## Entorno

Conda env `thesis` (`C:\Users\snowo\miniconda3\envs\thesis`) -- el único con
`torch<->numpy` funcionando + `deepchem` + PyG + rdkit + transformers + `h5py`.
(`envs\chem` tiene deepchem+h5py pero su `torch` tiene rota la interop con numpy.
`envs\thesis-train` no tiene deepchem. El env base de miniconda no tenía nada de esto.)

## Datos (NO están en git -- ver `.gitignore`)

`data/featurized_pool/`, `data/rederived_splits/` y `checkpoints/crossmodal/` quedan fuera
de git, igual que el resto de `data/`/`checkpoints/` en el proyecto. Si cloná el repo sin
copiarlos a mano, `data/featurized_pool/` se regenera sola en el primer run de
`benchmark/native_split.py`. (`data/molformer_embeddings/` ya no la usa nada en
`crossmodal_model` -- era exclusiva de la rama MolFormer/DNM eliminada.)

## Cómo correr

Regresión (Graph+SMILES-only), split oficial fijo, comparable con `fase2_baselines.csv`:
```
python crossmodal_model/benchmark/scaffold_fixed.py --datasets freesolv --seeds 2025 2026 2027                      # MoLA original
python crossmodal_model/benchmark/scaffold_fixed.py --datasets freesolv --seeds 2025 2026 2027 --positional-smiles   # fix A2
```

Generación condicionada, entrenamiento serio (100 epochs, patience 15, muestreo real):
```
python crossmodal_model/generation/train.py --dataset freesolv --epochs 100 --patience 15 --num-samples-per-mol 5
```

Chequeo semántico del conditioning (no reentrena, usa el checkpoint ya guardado):
```
python crossmodal_model/generation/eval_conditioning.py
```

Prueba ad-hoc con tu propia molécula:
```
python crossmodal_model/generation/from_smiles.py --smiles "TU_SMILES" --property-values -8 -3.8 2 --num-samples 8
```
