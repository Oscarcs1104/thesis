# Contexto del proyecto — generación molecular condicionada

## Estado actual

Repo con arquitectura HybridMoLA (encoder híbrido grafo + SMILES char, cabeza de
regresión, decoder Transformer sobre SELFIES). Último commit: "Failed attempts".

El condicionamiento actual es estilo Chemformer: un escalar de propiedad proyectado
por `property_proj` (Linear(1→H) → Tanh → Linear(H→H)) que se prepende como token en
posición 0 del `memory` sobre el que el decoder hace cross-attention.

```
memory = [ prop | átomo₁ … átomoₙ | char₁ … char_L ]     # [B, 1+N+L, H]
```

Dataset: FreeSolv (~642 moléculas).

## Diagnóstico: por qué no funciona

**1. El objetivo de entrenamiento es un autoencoder.**

En `train_hybrid.py`:
- el memory contiene la molécula semilla M (átomos + caracteres)
- el target del decoder es SELFIES(M) — la misma M
- el token de propiedad es `y = f(M)` — su propio valor verdadero

Por tanto `H(target | memory) = 0` antes de mirar el token de propiedad. El token no
aporta información incremental y el gradiente no tiene motivo para enseñar al decoder
a usarlo. Copiar vía cross-attention (patrón casi diagonal átomo_i → token_i) es
mucho más fácil de aprender.

Agravante: no hay cuello de botella. El memory es la secuencia completa sin poolear,
un vector por átomo y por carácter. Un autoencoder sin compresión.

**2. La `reg_loss` conjunta empeora el problema.**

Redundancia informacional no implica redundancia computacional — el decoder podría
usar el token como atajo barato si extraer `y` de los estados atómicos fuera costoso.
Pero la cabeza de regresión entrena justamente para que `y` sea linealmente legible
desde esos estados. El término añadido para compartir representación es el que
garantiza que el token no aporte nada ni siquiera en cómputo.

**3. Desajuste entrenamiento/inferencia.**

```
entrenamiento:  memory(M) + y = f(M)          # siempre consistentes
inferencia:     memory(M) + y = f(M) ± 1σ     # combinación nunca vista
```

Al pedir un shift no se pide algo que el modelo aprendió a hacer mal; se pide algo
fuera de distribución. La respuesta es indefinida.

**4. Datos insuficientes para el objetivo que sí tiene.**

Tanimoto a la semilla de 0.224 es incoherente con "copiador puro" (debería rondar
0.9). Un TransformerDecoder de 4 capas sobre vocabulario SELFIES desde cero con ~500
ejemplos no puede reconstruir. Chemformer preentrena sobre ~100M de ZINC.

Conclusión: objetivo que no enseña condicionamiento + datos insuficientes para
cumplir siquiera ese objetivo.

**5. Las métricas de evaluación están infladas.**

- El control direccional (93.8%) se mide con la propia cabeza de regresión del
  checkpoint. Encoder compartido, misma pérdida → circular. El docstring lo admite.
- Pearson(requested, estimated) = 0.684 es espurio: `requested = true_y + shift` y
  `estimated` correlaciona con `true_y` vía la semilla. Con condicionamiento
  estrictamente nulo el Pearson seguiría siendo alto solo por el componente `true_y`
  compartido. Lo válido es la correlación parcial controlando por `true_y`, o el
  efecto intra-semilla (Δestimated vs Δrequested para la misma semilla).
- `--prefix-len` (copiar los primeros N tokens SELFIES) es un parche, y depende del
  orden de canonicalización, no de la química.

## Plan de trabajo (72h, 1× RTX PRO 6000 Blackwell)

El cómputo no es el cuello de botella. Presupuesto ~40h de entrenamiento efectivo.
Con 2M moléculas y un modelo de ~30M params, 10 épocas son ~1.5h. Se prioriza tiempo
de iteración sobre tamaño de corpus.

### Bloque 0 (horas 0-2) — Asegurar el resultado negativo

`ablate_prop_token.py`: evaluar la CE de validación del checkpoint actual con el
`prop_token` sustituido por (a) ceros, (b) la media del batch, (c) un valor
aleatorio. Comparar ΔNLL contra el baseline.

Si ΔNLL ≈ 0 → el token se ignora, queda demostrado empíricamente.

Complemento: masa de atención sobre la posición 0 del memory, promediada sobre
capas/cabezas/posiciones target. Si ≈ 1/(1+N+L), está ignorado.

Esto convierte "Failed attempts" en un hallazgo con evidencia. Es entregable por sí
solo.

NOTA: no usar el probe (B) previsto en `probe_encoder.py` (generar con memory = solo
el token de propiedad) como diagnóstico. Es un desplazamiento de distribución brutal
para el decoder; la degradación no probaría que el encoder "manda", solo que el
decoder nunca vio esa entrada.

### Bloque 1 (horas 2-6) — Datos

Corpus: **MOSES (~1.9M) o GuacaMol (~1.6M)**. Elegidos por estar curados y tener
baselines publicados (validez, unicidad, novedad, FCD), no por tamaño.

Pipeline: SMILES → canonicalizar (RDKit) → SELFIES → vocabulario → tensor int16
cacheado en `.npy`. Usar `multiprocessing` sobre todos los cores: en un solo proceso
esto tarda horas, en paralelo minutos. Cachear y no volver a tocarlo.

Etiquetas: calcular con RDKit para cada molécula `logP` (Crippen), `TPSA`, `QED`,
`MW`. Oráculo exacto, sin ruido de etiqueta, ilimitado.

### Bloque 2 (horas 6-20) — Pretrain condicional

**Tirar el encoder. Decoder-only sobre SELFIES con propiedades como tokens de
prefijo discretizados:**

```
[logP_bin7] [TPSA_bin3] [QED_bin9] [BOS] tok₁ tok₂ … [EOS]
```

Cada propiedad bineada en ~20 cuantiles; cada bin es un embedding aprendido.
Pérdida de LM estándar sobre los tokens SELFIES.

Por qué esto sí funciona: **la molécula target ya no está en el condicionamiento**.
El prefijo es lo único que determina qué generar, luego `H(target | prefijo) > 0` y
el gradiente está obligado a usarlo. No hay atajo de copiado porque no hay nada que
copiar.

Detalles:
- Dropout del prefijo al 15% con token `[uncond]` aprendido → habilita
  classifier-free guidance en muestreo. Es la perilla de intensidad de control y da
  un barrido de resultados con un solo modelo entrenado. Diagnóstico gratis: si `w`
  grande no cambia nada, la condición no se usa.
- 8-12 capas, d_model=512, 8 cabezas ≈ 25-40M params.
- BF16, `torch.compile`, SDPA, AdamW fusionado, batches ordenados por longitud.
- Batch ~1024 secuencias (con 96GB cabe más, pero no compensa).
- Checkpoints cada 30 min. La entrega se pierde por un OOM a la hora 19, no por
  falta de FLOPs.

Si por algún motivo se conserva la arquitectura encoder-decoder, aplicar en este
orden de rendimiento esperado:
1. **FiLM/AdaLN** en vez de token prependido — modular escala y sesgo de cada capa
   del decoder con la propiedad. No se puede rodear por atención; multiplica todo el
   forward. Un token entre ~130 posiciones es trivial de ignorar.
2. **Features de Fourier** del escalar antes de proyectar (como timestep embeddings
   de difusión). Un escalar crudo en `Linear(1→H)` da gradiente pobre y señal de baja
   frecuencia. Estandarizar la entrada — ahora entra en kcal/mol crudos.
3. **Classifier-free guidance**, como arriba.

### Bloque 3 (horas 20-28) — Eval con oráculo exacto

Pedir logP en el bin *k*, generar 1000 moléculas, medir el logP real con RDKit. Sin
predictor circular, sin OOD, sin ruido de etiqueta.

Métricas: curva pedido-vs-obtenido, MAE en unidades reales, validez, unicidad,
novedad, y barrido de la escala `w` de CFG.

**Este es el entregable principal.** Resultado limpio y defendible por sí solo.

### Bloque 4 (horas 28-48) — FreeSolv

Fine-tuning del mismo modelo añadiendo un token de prefijo de solvatación, con las
642 moléculas. El generador ya sabe química; solo aprende a leer un condicionamiento
más.

Si funciona, bien. Si no, los resultados del bloque 3 están intactos y el
fine-tuning se reporta como limitación por escasez de etiquetas — conclusión honesta
y correcta.

No validar nunca con la propia cabeza de regresión del checkpoint. Y tener presente
que un predictor independiente entrenado con 500 moléculas tampoco es evidencia
sobre moléculas OOD.

### Bloque 5 (horas 48-72) — Escritura, figuras, colchón

No negociable. Se va la mitad en esto.

## Regla operativa

Cada bloque termina en algo entregable. A la hora 28 hay un resultado completo aunque
FreeSolv falle entero. Nunca estar a una sola jugada de no tener nada.

## Alternativas consideradas (por si el plan principal se cae)

- **Autoencoder no condicionado + predictor sobre el latente + optimización en
  espacio latente.** El generador aprende de 10⁵-10⁶ moléculas sin etiquetas; la
  escasez de etiquetas solo afecta al predictor, que es la parte que tolera pocos
  datos.
- **Búsqueda (GraphGA / REINVENT) usando el regresor como función de puntuación.**
  Más simple, y una baseline sorprendentemente dura.
- Si se quiere conservar la similitud a la semilla, usar **scaffold de Murcko o un
  fragmento fijo** como restricción, no un prefijo de tokens SELFIES canónicos.

## Convenciones del repo

- `.gitignore` excluye `results/mola/*_samples.csv` y `*_conditioning_semantic.csv`.
  Los únicos números disponibles del trabajo previo están en los docstrings de
  `train_hybrid.py` y `eval_hybrid_control.py`.
- Dos vocabularios distintos: `char_vocab` (entrada encoder, SMILES char a char,
  padded a 100) y `selfies_vocab` (salida decoder). SELFIES en la salida es
  deliberado: `selfies.decoder()` no puede producir un SMILES inválido desde una
  cadena bien formada → validez por construcción.
