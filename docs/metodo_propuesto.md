# Método propuesto

> **Estado:** borrador técnico completo, prosa revisada.
> **Destinatario:** tribunal y revisores de la memoria de tesis.
> **Todas las cifras proceden de ejecuciones reales del pipeline sobre MOSES completo.**
> No inventar, redondear ni sustituir ningún número de este documento.

---

## 1. Planteamiento

La hipótesis de trabajo es que una representación molecular que combina el grafo y la
cadena SMILES describe una molécula mejor que cualquiera de las dos modalidades por
separado. La contribución no es una arquitectura nueva, sino la comprobación de esa
hipótesis **en dos tareas distintas con un único eje de ablación**: leyendo moléculas
(regresión de propiedades) y escribiéndolas (generación condicionada).

Ese eje no cambia entre tareas —activar o desactivar cada rama del encoder—, de manera que
las dos mitades del trabajo responden a la misma pregunta con evidencia independiente. El
hallazgo se replica en dos condiciones en lugar de descansar sobre un único experimento.

```
  §2–§4  datos          MOSES · etiquetas RDKit · pares análogos
     │
  §5–§6  encoder        HybridMoLA, preentrenado sobre 1,94 M moléculas
     │
     ├── §7  tarea A    fine-tuning   ESOL · FreeSolv · Lipophilicity
     └── §8  tarea B    decoder SELFIES ── generación condicionada
```

Un único checkpoint del preentrenamiento alimenta tanto el fine-tuning como el decoder. Es
la única forma de que las dos mitades compartan pesos, y no solo arquitectura.

Las secciones que siguen recorren el pipeline en ese orden: corpus, etiquetas y pares de
entrenamiento; arquitectura y preentrenamiento del encoder; y las dos tareas sobre las que
se mide.

---

## 2. Corpus de preentrenamiento

Se emplea **MOSES** (Molecular Sets), un subconjunto curado de ZINC restringido a espacio
químico *drug-like*. Se elige frente a volcados mayores por estar depurado y por disponer
de líneas base publicadas de validez, unicidad, novedad y FCD, lo que permite situar los
resultados generativos frente a la literatura.

### 2.1 Canonicalización y codificación

Cada SMILES atraviesa cuatro pasos en RDKit: se parsea a molécula, se recanonicaliza
(`Chem.MolToSmiles`), se calcula su InChIKey como identificador estructural y se convierte
a **SELFIES** mediante `selfies.encoder`. Las moléculas que RDKit rechaza, o que SELFIES no
puede representar, se descartan.

La elección de SELFIES como espacio de salida del decoder es deliberada: `selfies.decoder`
no puede producir un SMILES inválido a partir de una cadena bien formada, de modo que la
validez química queda garantizada por construcción y no depende de que el modelo la
aprenda.

### 2.2 Vocabulario

El alfabeto SELFIES resultante consta de **29 tokens** (25 de SELFIES más `<PAD>`,
`<START>`, `<END>`, `<OTHER>`). Esa compacidad no es un error: MOSES está filtrado a un
espacio químico estrecho y su alfabeto SELFIES es genuinamente pequeño.

> **Garantía · validez por construcción.** No se aplica poda de tokens por frecuencia.
> Colapsar tokens raros en `<OTHER>` es la práctica habitual en modelado de lenguaje, pero
> aquí rompería la garantía de validez: `<OTHER>` no es decodificable, y una secuencia
> generada que lo contenga no es una molécula. Si se fija un umbral de frecuencia, se
> descartan las moléculas afectadas en lugar de corromperlas, de modo que todo
> identificador del vocabulario corresponde a SELFIES real.

Las secuencias se truncan al percentil 99,5 de longitud, **45 tokens** (mediana 35), y se
almacenan como una matriz `int16`. `<START>` y `<END>` se añaden en tiempo de entrenamiento
y no se almacenan.

### 2.3 Control de fuga hacia los conjuntos de evaluación

Toda molécula del corpus cuyo InChIKey coincida con una del *split de test* de ESOL,
FreeSolv o Lipophilicity se elimina. Se usa InChIKey y no SMILES crudo para que una
escritura distinta de la misma molécula también se detecte.

| Conjunto | Moléculas en test | Eliminadas del corpus |
|---|---:|---:|
| ESOL | 112 | 11 |
| FreeSolv | 65 | 0 |
| Lipophilicity | 420 | 27 |
| **Total (InChIKeys únicos)** | **591** | **38** |

La fila de total cuenta **InChIKeys únicos**, no la suma de las tres filas: 597 − 591 = 6
moléculas figuran en el conjunto de test de más de un dataset, y basta una coincidencia
para excluirlas del corpus. La columna de eliminadas sí es aditiva.

El patrón es químicamente coherente: FreeSolv son disolventes pequeños y no se solapa con
un corpus drug-like, mientras que Lipophilicity procede de ChEMBL y aporta la mayoría de
las coincidencias.

El solape de *scaffolds* se mide y se reporta, pero **no** se elimina: 184 esqueletos de
Murcko compartidos, que cubren 72 092 moléculas del corpus (**3,72 %**). Eliminar todo
scaffold compartido arrancaría quimiotipos completos de un corpus drug-like, y ningún
protocolo publicado de preentrenamiento lo hace; el número queda constando para que el
lector juzgue su alcance.

Tras estos filtros el corpus queda en **1 936 539 moléculas**.

---

## 3. Etiquetado con oráculo exacto

Depurado el corpus, falta dotarlo de señal supervisada. Para cada molécula se calculan
cuatro descriptores con RDKit:

| Propiedad | Función | Interpretación |
|---|---|---|
| logP | `Crippen.MolLogP` | coeficiente de reparto octanol/agua |
| TPSA | `rdMolDescriptors.CalcTPSA` | superficie polar topológica (Å²) |
| QED | `QED.qed` | estimación cuantitativa de *drug-likeness*, en [0, 1] |
| MW | `Descriptors.MolWt` | peso molecular (Da) |

Estos descriptores son un **oráculo**, no un modelo: son funciones deterministas de la
estructura. De ahí se derivan tres propiedades metodológicas que el diseño explota:

- **No hay ruido de etiqueta**, a diferencia de una medida experimental.
- **No hay techo impuesto por un maestro**, a diferencia de pseudo-etiquetar con un
  predictor entrenado.
- **La evaluación no es circular**: en la generación condicionada se pide un objetivo y se
  mide lo obtenido con la misma función, sin ningún predictor aprendido en el bucle.

La alineación con las tareas finales tampoco es accidental. El logP de Crippen es el
término dominante de la ecuación de Delaney para solubilidad acuosa (ESOL), y el logD 7.4
de Lipophilicity es logP corregido por ionización. El preentrenamiento no resuelve, por
tanto, una tarea auxiliar arbitraria, sino una versión exacta y abundante de la tarea de
destino.

Las moléculas cuyo cálculo falla se conservan con etiqueta `NaN` en lugar de eliminarse,
para que la fila *i* de la matriz de etiquetas siga correspondiendo a la fila *i* del
corpus; se filtran aguas abajo.

---

## 4. Minado de pares análogos

Esta etapa construye la señal de entrenamiento de la mitad generativa y es la que resuelve
el problema central del diseño previo.

### 4.1 El problema que resuelve

Un generador condicionado que se entrena sobre `(M → M, y = f(M))` tiene la molécula
objetivo dentro del propio *memory* del decoder. Reproducirla no requiere leer el token de
condición: `H(objetivo | memory) = 0` antes siquiera de mirarlo, y el gradiente no tiene
motivo para enseñar al decoder a usarlo. El condicionamiento resultante es nominal.

La solución es que la molécula objetivo **no** sea la de entrada:

```
entrada al encoder   M_a   Clc1ccccc1   clorobenceno    logP 2,84
condición            Δ     −0,57
objetivo del decoder M_b   Fc1ccccc1    fluorobenceno   logP 2,27
```

M_b no está en el *memory*. La única indicación de que hay que sustituir el cloro por
flúor, y no por bromo, es el Δ, con lo que el gradiente queda obligado a usarlo.

Este planteamiento elimina además el desajuste entre entrenamiento e inferencia. Con
condicionamiento absoluto se pedía `y = y_real ± 1σ`, una combinación que jamás aparece en
los datos; con un Δ relativo, pedir `Δ = −1` en inferencia es exactamente lo que el modelo
vio miles de veces.

### 4.2 Procedimiento

1. **Agrupación por scaffold de Murcko.** Comparar 1,94 M × 1,94 M moléculas es inviable.
   Los análogos se concentran, casi por definición, entre moléculas que comparten
   esqueleto, de modo que agrupar convierte un problema cuadrático en uno lineal. El
   scaffold es únicamente un mecanismo de búsqueda: **nunca entra en la red**.
2. **Muestreo de candidatos.** Dentro de cada grupo se comparan hasta 200 candidatos por
   molécula en lugar del grupo entero, lo que acota el coste con independencia de lo
   poblado que esté un scaffold. Los grupos de más de 5 000 miembros se parten en bloques
   para equilibrar la carga.
3. **Banda de similitud.** Se aceptan pares con Tanimoto sobre huellas ECFP4 en
   `[0,50 , 0,95]`. Por debajo de 0,50 no hay una modificación sino otra molécula, y el
   encoder no puede ayudar; por encima de 0,95 hay una copia, que no enseña nada.
4. **Preferencia por el análogo más lejano.** Cuando hay más candidatos admisibles que el
   tope de 10 por molécula, se retienen los *menos* similares: son los que conllevan el
   mayor cambio estructural y, por tanto, el mayor Δ que la condición debe explicar.
5. **Ambas direcciones.** De cada par hallado se emiten `(a, b)` y `(b, a)`. El Δ es una
   magnitud con signo, y entrenar en una sola dirección sesgaría cada token de condición
   hacia un signo.
6. **Pares identidad.** Un 5 % de los ejemplos son `(M, M)` con Δ = 0. Fijan la semántica
   de «no cambies nada» y proporcionan una prueba de sanidad en evaluación: pedir Δ = 0
   debe devolver la semilla.

El resultado son **12 991 314 pares**, de los cuales 649 566 son de identidad.

| Propiedad | desv. típica de Δ | p1 | p99 | \|Δ\| > 1 σ |
|---|---:|---:|---:|---:|
| logP | 0,685 | −1,766 | +1,766 | 27,7 % |
| TPSA | 15,784 | −43,09 | +43,09 | 29,4 % |
| QED | 0,066 | −0,198 | +0,198 | 21,9 % |
| MW | 27,679 | −69,09 | +69,09 | 32,1 % |

La media es exactamente cero y los percentiles son simétricos, lo que confirma que la
emisión bidireccional funciona. La última columna usa unidades de desviación típica porque
un umbral absoluto carece de sentido comparativo: «más de 1» es trivial para MW en daltons
e inalcanzable para QED, cuyo rango completo es [0, 1].

El percentil 99 del Δ de logP, **±1,77**, define el rango de control que la evaluación
puede solicitar. Pedir desplazamientos mayores sería extrapolación, precisamente el defecto
que este diseño corrige.

---

## 5. Arquitectura del encoder

El encoder, **HybridMoLA**, tiene dos ramas y un mecanismo de fusión entre capas. Con
`hidden = 256` y 3 capas suma aproximadamente **4,7 M parámetros**.

### 5.1 Rama de grafo

La molécula se representa como grafo con featurización categórica estilo OGB: **9 columnas
por átomo** y **3 por enlace**, cada una un índice a su propio `nn.Embedding`.

| Nivel | Campos |
|---|---|
| Átomo | `atomic_num`, `chirality`, `degree`, `formal_charge`, `num_hs`, `num_radical_electrons`, `hybridization`, `is_aromatic`, `is_in_ring` |
| Enlace | `bond_type`, `stereo`, `is_conjugated` |

La alternativa habitual —un vector denso de reales— trataría el número atómico como
magnitud continua y haría que el modelo heredara la relación «el carbono está numéricamente
cerca del nitrógeno y lejos del azufre», que no significa nada químicamente. Con embeddings
por columna, en cambio, el modelo sitúa cada tipo donde le resulte útil.

Sobre esa representación operan capas **GINEConv**, variante de *Graph Isomorphism Network*
que incorpora características de arista: cada capa suma el embedding del enlace al mensaje
del vecino antes del MLP, en lugar de agregar vecinos de forma ciega al tipo de enlace.
Tras cada capa se aplica ReLU, *dropout* y una operación de *pooling* configurable (suma
por defecto), que produce un vector por grafo **en cada capa**.

### 5.2 Rama de SMILES

La cadena SMILES se tokeniza a nivel de carácter, se acolcha a 100 posiciones y se le suma
un *positional embedding*. Cada capa es un `TransformerEncoderLayer` con 8 cabezas y
máscara de acolchado. El estado se reduce a un vector por molécula mediante media
enmascarada sobre los caracteres reales.

El acolchado y las posiciones son obligatorios aquí: un encoder de caracteres invariante a
permutación y diluido por padding no puede sostener generación coherente.

### 5.3 Fusión entre capas (MoLA)

La fusión no combina únicamente la salida final de cada rama. Cada capa de cada modalidad
aporta un token; todos se apilan y se someten a *self-attention* con 8 cabezas, y el
resultado se reduce con una suma ponderada por pesos aprendidos, uno por token. La
operación es *self-attention*: el conjunto de tokens se atiende contra sí mismo, con
consulta, clave y valor idénticos. En el código de MoLA el módulo se llama
`cross_attention` porque cruza modalidades y profundidades, pero formalmente es
auto-atención sobre el conjunto apilado.

```
capa 1    [ token_grafo₁ , token_smiles₁ ]  ┐
capa 2    [ token_grafo₂ , token_smiles₂ ]  ├─ self-attention ── suma ponderada
capa 3    [ token_grafo₃ , token_smiles₃ ]  ┘   (8 cabezas)      (pesos aprendidos)
```

Con `use_graph` o `use_smiles` desactivados, el número de tokens por capa baja de 2 a 1 y
el vector de pesos se dimensiona en consecuencia: una rama apagada no aporta parámetros, en
lugar de aportar ceros que se seguirían contando y entrenando.

La premisa de esta fusión no es solo que capas distintas codifiquen información distinta,
sino que **el objetivo de entrenamiento las moldea** para que así sea. Esa segunda mitad de
la premisa exige que las ramas se entrenen conjuntamente con la fusión, y es la razón por
la que el preentrenamiento descrito a continuación no congela nada.

---

## 6. Preentrenamiento supervisado

El encoder se entrena sobre el corpus completo con una cabeza de regresión multitarea que
predice las cuatro etiquetas de RDKit simultáneamente. El interés no está en esos cuatro
valores, sino en la representación que el encoder desarrolla al producirlos.

### 6.1 Estandarización de objetivos

Las cuatro propiedades se estandarizan con la media y la desviación del conjunto de
entrenamiento. Sin ello, un error cuadrático medio sin ponderar estaría dominado por el
peso molecular, cuya desviación típica se mide en decenas de daltons, frente al QED, cuya
escala completa es el intervalo [0, 1]; el modelo resultante sería un regresor de MW con
tres adornos. Los valores exactos de media y desviación por propiedad quedan registrados
en `labels_meta.json` y se reportan junto a los resultados. El RMSE de validación se reporta de
vuelta en las unidades propias de cada propiedad.

### 6.2 Configuración

| Parámetro | Valor | Justificación |
|---|---|---|
| Pasos | 40 000 | presupuesto fijo, idéntico entre configuraciones |
| Tamaño de lote | 256 | — |
| Optimizador | AdamW fusionado | — |
| Planificación | coseno tras 1 000 pasos de calentamiento | — |
| Precisión | bf16 (pérdida en fp32) | — |
| Partición de validación | 1 % | aleatoria; el test ya se excluyó del corpus |

Como todos los pesos son entrenables desde una inicialización aleatoria, se emplea una
única tasa de aprendizaje. Distinguir entre tasa del cuerpo y tasa de la cabeza solo tiene
sentido cuando hay un *backbone* preentrenado que proteger.

---

## 7. Predicción de propiedades experimentales

Concluido el preentrenamiento, la primera mitad del trabajo confronta lo aprendido con
medidas experimentales. El encoder se evalúa sobre tres conjuntos de regresión de
MoleculeNet, con la cabeza de regresión reinicializada a una única salida.

| Conjunto | Propiedad | Moléculas | Test |
|---|---|---:|---:|
| ESOL (Delaney) | solubilidad acuosa, log mol/L | ≈ 1 120 | 112 |
| FreeSolv (SAMPL) | energía libre de hidratación, kcal/mol | ≈ 650 | 65 |
| Lipophilicity | logD a pH 7,4 | ≈ 4 200 | 420 |

Los CSV se descargan del repositorio público de MoleculeNet, se canonicalizan, se
deduplican por InChIKey —promediando el valor cuando dos entradas son la misma molécula— y
se parten con un **split por scaffold de Murcko fijo** (80/10/10, semilla 2025). El split
se congela en disco y lo comparten todas las configuraciones y semillas, de modo que
ninguna diferencia entre filas de la tabla pueda atribuirse a una partición distinta.

### 7.1 El contraste que se mide

Cada configuración se entrena dos veces, idénticas salvo en el origen de los pesos:

- **Sin preentrenar** — inicialización aleatoria; el modelo solo ve las 650 a 4 200
  moléculas del conjunto.
- **Preentrenada** — parte del checkpoint del preentrenamiento, descartando su cabeza.

La primera fila no es prescindible. «RMSE 0,85 con preentrenamiento» no significa nada por
sí solo: el resultado es la diferencia entre ambas.

> **Garantía · consistencia del vocabulario.** Al inicializar desde el checkpoint, el
> vocabulario de caracteres se toma del propio checkpoint y no se reconstruye a partir del
> conjunto pequeño. El *embedding* de SMILES está indexado por identificador de carácter:
> reconstruir el vocabulario haría que cada fila de esa matriz preentrenada pasara a
> representar un carácter distinto. El modelo entrenaría, convergería, y estaría mal, sin
> emitir ningún error.

Se reportan RMSE (métrica principal, en las unidades del objetivo), MAE, R² y NRMSE, esta
última normalizada por la desviación típica del conjunto de entrenamiento —y no por el
rango— para que un único valor extremo en un test pequeño como el de FreeSolv no la
distorsione. Cada celda se repite con tres semillas (2025, 2026, 2027) y se reporta media y
desviación.

---

## 8. Generación condicionada

La segunda mitad del trabajo usa el mismo encoder para escribir moléculas en vez de
leerlas. La tarea es `p(M_b | M_a, Δpropiedad)`: dada una molécula de partida y un
desplazamiento objetivo, producir una molécula análoga que lo satisfaga. Es generación
condicionada en la que una de las condiciones es una molécula; en la literatura de diseño
de fármacos corresponde a optimización de *leads*.

### 8.1 Discretización de la condición

Cada uno de los cuatro Δ se discretiza en **20 bins por cuantiles**, más un bin nulo, y
cada bin tiene un *embedding* aprendido. Los bordes se ajustan **solo con los pares de
entrenamiento**.

La discretización sustituye a la proyección de un escalar crudo. Un número aislado a través
de un `Linear(1→H)` produce una señal de baja frecuencia y gradiente pobre; en el diseño
previo, además, entraba sin estandarizar, en kcal/mol. Un índice de bin, en cambio,
selecciona un vector de rango completo. Bins y *features* de Fourier resuelven el mismo
problema, por lo que son alternativas y no se acumulan.

Los cuantiles se colapsan cuando la distribución es picuda —como ocurre con los Δ
acumulados en cero—; los bordes duplicados se funden para no crear bins inalcanzables cuyas
filas del embedding se entrenarían con cero ejemplos.

### 8.2 Composición del memory

```
memory = [ Δlogp │ Δtpsa │ Δqed │ Δmw │ átomo₁…átomoₙ │ char₁…char_L ]
           └────── condición ──────┘   └──── M_a, lo que hay que modificar ────┘
```

Los estados del encoder entran sin poolear: un vector por átomo y uno por carácter, no un
único vector resumen. El decoder hace *cross-attention* sobre toda la secuencia.

Se conserva el token prependido en lugar de recurrir a FiLM o AdaLN, porque el token se
ignoraba en el diseño previo por ser **redundante**, no por ser un token; eliminada la
redundancia, el mecanismo más simple debería bastar. Los diagnósticos de §8.5 determinan si
es así, y FiLM queda como alternativa documentada si no lo es.

### 8.3 Decoder

`nn.TransformerDecoder` estándar: *self-attention* causal sobre los tokens ya emitidos,
*cross-attention* sobre el *memory* con máscara de acolchado, y pérdida de entropía cruzada
por token sobre los SELFIES de M_b, con *teacher forcing*. El decoder tiene 6 capas y se
entrena 60 000 pasos con lote 256.

El encoder conserva exactamente la configuración con la que fue preentrenado —`hidden = 256`
y 3 capas—, y no podría ser de otro modo: unos pesos preentrenados solo encajan en la forma
en que se entrenaron. El ancho del decoder queda fijado por el mismo valor, porque
*cross-attiende* sobre el *memory* que produce el encoder. En la implementación, los scripts
de fine-tuning y de generación leen esas dimensiones del propio checkpoint en lugar de
tomarlas de sus valores por defecto, de modo que la cadena permanece consistente aunque el
preentrenamiento se repita con otro tamaño.

El gradiente retrocede por la *cross-attention*, atraviesa el *memory* y alcanza tanto los
embeddings de condición como el encoder. El encoder queda así entrenado también por la
señal generativa, que es lo que permite que desactivar una de sus ramas degrade la
generación de forma medible.

### 8.4 Dropout de condición y guía sin clasificador

Durante el entrenamiento, cada propiedad se sustituye por su bin nulo con probabilidad
0,15, **de forma independiente** entre propiedades. Esa independencia permite que una
petición nombre una propiedad y deje las demás sin especificar —«Δ logP = +1, el resto me
da igual»—, que es la petición realista; un *dropout* conjunto solo permitiría
condicionamiento de todo o nada.

Eso habilita *classifier-free guidance* en muestreo:

```
logits = logits_sin_condición + w · (logits_con_condición − logits_sin_condición)
```

El peso `w` regula la intensidad del control y permite un barrido completo de resultados
con un solo modelo entrenado. Es además un diagnóstico sin coste adicional de
entrenamiento: **si aumentar `w` no cambia nada, la condición no se está utilizando.**

### 8.5 Diagnósticos de uso de la condición

El fallo del diseño previo no fue que el condicionamiento funcionara mal, sino que no
existía sin que nada lo delatara: la pérdida bajaba, las moléculas eran válidas, y el token
simplemente no se leía. Por eso el método incorpora tres comprobaciones cuyo único
propósito es detectar esa situación, independientes de la calidad de lo generado.

**Barrido del peso de guía.** Se muestrea el mismo conjunto de semillas y condiciones con
`w` creciente. Si el control de propiedad no mejora al aumentar `w`, la condición no está
interviniendo en la predicción. No requiere reentrenar y su coste es una pasada adicional
por muestreo.

**ΔNLL con la condición alterada.** Se recalcula la entropía cruzada de validación
sustituyendo los tokens de condición por variantes que no llevan información sobre el par:
bin nulo, bins aleatorios, y —la prueba decisiva— los bins **barajados dentro del lote**,
que conservan exactamente la distribución marginal y solo rompen el emparejamiento entre
molécula y Δ. Se reporta la diferencia respecto de la condición correcta con un intervalo
de confianza *bootstrap* emparejado por molécula. Si ese intervalo contiene el cero, la
condición no aporta información incremental y el condicionamiento es nominal. El intervalo
es imprescindible: sin él, un ΔNLL pequeño no se distingue del ruido.

**Masa de atención sobre las posiciones de condición.** Se registran los pesos de
*cross-attention* del decoder y se promedia, sobre capas, cabezas y posiciones objetivo no
acolchadas, la fracción que recae sobre los cuatro tokens de condición. La referencia es la
masa uniforme, 4/(4 + N + L): si la observada no la supera, el decoder no los mira. El mismo
registro desglosa la masa entre nodos de grafo y caracteres SMILES, lo que indica sobre qué
rama del encoder se apoya realmente el decoder.

Las dos últimas están implementadas sobre el diseño anterior en
`crossmodal_model/generation/ablate_prop_token.py`, donde sirvieron para documentar por qué
aquel condicionamiento no funcionaba; el procedimiento se traslada sin cambios al esquema de
bins sustituyendo el escalar por los cuatro tokens.

---

## 9. Protocolo experimental

### 9.1 Partición de los pares

Los pares se parten **por scaffold**, no por par. Como el minado opera dentro de grupos de
scaffold, asignar esqueletos completos a una partición mantiene todos los pares íntegros y
garantiza que ninguna molécula aparezca en dos particiones.

> **Garantía · fuga entre particiones.** Una partición por par colocaría la misma molécula
> en entrenamiento y en test, como origen en una y como destino en la otra, y las cifras
> retenidas medirían memorización. Verificado sobre datos sintéticos minados con el mismo
> procedimiento: la partición por scaffold deja el 100 % de los pares íntegros y los
> conjuntos de moléculas disjuntos, mientras que una partición por par de los mismos datos
> filtraría decenas de miles de moléculas a ambos lados.

### 9.2 Ablación

Tres brazos, idénticos salvo en las ramas activas del encoder: **grafo + SMILES**, **solo
grafo**, **solo SMILES**. Se ejecutan en las dos mitades del trabajo.

El presupuesto se fija en **pasos**, no en épocas. Los brazos se ejecutan secuencialmente
sobre una única GPU, y un brazo que entrenara más tiempo por tener épocas más baratas
convertiría la comparación en una sobre tiempo de reloj en lugar de sobre las modalidades.

### 9.3 Evaluación generativa con oráculo

El protocolo elimina la circularidad de raíz:

1. Se toman moléculas semilla del **split de test**, es decir, pertenecientes a scaffolds
   que el modelo no vio en ningún par de entrenamiento. Por la partición descrita en 9.1,
   ninguna de esas moléculas apareció como origen ni como destino.
2. Se solicita un Δ dentro del rango que los datos soportan, derivado de los percentiles de
   la tabla de §4.
3. Se muestrea y se decodifica SELFIES a SMILES.
4. **Se mide la propiedad real de lo generado con RDKit** y se compara con lo solicitado.

El contraste con la evaluación anterior es directo: aquella puntuaba las moléculas
generadas con la cabeza de regresión del propio checkpoint generativo —encoder compartido y
misma pérdida, es decir, el modelo evaluándose a sí mismo—, mientras que RDKit no comparte
nada con el modelo.

Métricas: tasa de acierto (fracción de generadas cuyo Δ real cae en el bin solicitado), MAE
entre Δ pedido y obtenido, curva pedido-vs-obtenido —cuya pendiente es la fuerza del
condicionamiento—, validez, unicidad, novedad, FCD, similitud de Tanimoto a la semilla
(alta pero inferior a 1; si es 1, el modelo copia), y barrido del peso de guía.

---

## 10. Limitaciones

- **Desplazamiento de dominio en el preentrenamiento.** MOSES está filtrado a espacio
  drug-like, mientras que FreeSolv son mayoritariamente disolventes pequeños —el mismo
  contraste que ya explicaba las cero coincidencias de la sección 2.3—. La distribución de
  preentrenamiento cubre mal ese conjunto y es donde cabe esperar menor transferencia.
- **No hay generación *de novo*.** El encoder necesita una entrada, de modo que no es
  posible generar sin molécula de partida. Es la contrapartida directa de exigir que el
  encoder sea imprescindible: la generación *de novo* carece de encoder y, por tanto, de
  fusión multimodal que ablatar. En optimización de *leads*, que es el caso de uso al que
  corresponde la tarea, siempre existe una molécula de partida.
- **Rango de control acotado por los datos.** El percentil 99 del Δ de logP es ±1,77;
  solicitar desplazamientos mayores sería extrapolación.
- **Posible dominancia de una modalidad.** Al entrenar ambas ramas conjuntamente existe la
  posibilidad real de que la de SMILES asuma todo el trabajo. Sería un resultado negativo
  para la hipótesis, y es precisamente lo que la ablación de tres brazos está construida
  para detectar en lugar de ocultar.

---

## Pendiente de completar

Todo lo que dependa de resultados, que aún no existen:

- Tabla de RMSE / MAE / R² por conjunto, configuración y semilla, con y sin
  preentrenamiento.
- Curva pedido-vs-obtenido de la generación, y su pendiente, para los tres brazos.
- Barrido del peso de guía.
- Tamaños exactos de ESOL, FreeSolv y Lipophilicity tras la deduplicación por InChIKey
  (en el documento figuran como aproximados).

---
