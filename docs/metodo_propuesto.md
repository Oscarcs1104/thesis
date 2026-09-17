# Método propuesto

> **Estado:** borrador técnico completo, prosa revisada.
> **Destinatario:** tribunal y revisores de la memoria de tesis.
> **Todas las cifras proceden de ejecuciones reales del pipeline sobre MOSES completo.**
> No inventar, redondear ni sustituir ningún número de este documento.

---

## 1. Método Propuesto

### 1.1 Planteamiento e hipótesis

La hipótesis de trabajo sostiene que una representación molecular que combina el grafo y la
cadena SMILES describe una molécula mejor que cualquiera de las dos modalidades por
separado. La contribución no reside, por tanto, en una arquitectura nueva, sino en la
comprobación de esa hipótesis **en dos tareas distintas con un único eje de ablación**:
leyendo moléculas (regresión de propiedades) y escribiéndolas (generación condicionada).

Dado que ese eje —activar o desactivar cada rama del encoder— permanece invariante entre
tareas, las dos mitades del trabajo responden a la misma pregunta con evidencia
independiente, de modo que el hallazgo se replica en dos condiciones en lugar de descansar
sobre un único experimento.

```
  §2     datos          MOSES · etiquetas RDKit · pares análogos
     │
  §1.2   encoder        HybridMoLA, preentrenado sobre 1,94 M moléculas
     │
     ├── §3.2  tarea A   fine-tuning   ESOL · FreeSolv · Lipophilicity
     └── §3.3  tarea B   decoder SELFIES ── generación condicionada
```

Un único checkpoint del preentrenamiento alimenta tanto el fine-tuning como el decoder, ya
que es la única forma de que las dos mitades compartan pesos y no solo arquitectura.

Las tres subsecciones siguientes describen, en este orden, los componentes del sistema
(§1), los datos sobre los que opera (§2) y el procedimiento que lo entrena y lo evalúa
(§3).

### 1.2 Arquitectura del encoder: HybridMoLA

El encoder, **HybridMoLA**, se compone de dos ramas y un mecanismo de fusión entre capas.
Con `hidden = 256` y 3 capas suma aproximadamente **4,7 M parámetros**.

#### 1.2.1 Rama de grafo

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

#### 1.2.2 Rama de SMILES

La cadena SMILES se tokeniza a nivel de carácter, se acolcha a 100 posiciones y se le suma
un *positional embedding*. Cada capa es un `TransformerEncoderLayer` con 8 cabezas y
máscara de acolchado. El estado se reduce a un vector por molécula mediante media
enmascarada sobre los caracteres reales.

A diferencia de la rama anterior, el acolchado y las posiciones resultan aquí obligatorios:
un encoder de caracteres invariante a permutación y diluido por padding no puede sostener
generación coherente.

#### 1.2.3 Equiparación de capacidad entre las dos ramas

Una ablación de modalidades solo mide modalidades si las ramas tienen capacidad comparable.
No era el caso en la configuración por defecto. `nn.TransformerEncoderLayer` fija
`dim_feedforward = 2048` sin escalarlo con `d_model`, de modo que con un ancho oculto de 256
la rama de SMILES aplicaba una expansión de 8× en su bloque *feedforward*, mientras que el
MLP de actualización del GINEConv era `hidden → hidden`, expansión de 1×. El resultado:

| Rama | Parámetros (3 capas, ancho 256) |
|---|---:|
| SMILES | 3 945 216 |
| Grafo | 450 816 |

Un factor de **8,75×**. Bajo esa configuración, «solo SMILES supera a solo grafo» sería una
afirmación sobre capacidad, no sobre modalidad, y el brazo fusionado concentraría el 96 % de
sus parámetros en una sola rama, lo que impediría separar «la fusión aporta» de «una de las
dos ramas es mucho mayor».

Se introduce por ello un multiplicador de anchura interna en el MLP del GINEConv,
`hidden → 8·hidden → hidden`, que le da a la rama de grafo el mismo bloque *feedforward* que
la de SMILES. La razón cae a **1,23×**. Lo que resta es el bloque de atención, cuya
contrapartida en la rama de grafo es el paso de mensajes y no parámetros; no es un desajuste
que más anchura corrija, y forzar la igualdad exacta exigiría un multiplicador elegido para
cuadrar una cifra en lugar de por una razón arquitectónica.

El multiplicador viaja dentro del *checkpoint* junto al ancho oculto y el número de capas: los
pesos preentrenados solo encajan en la forma en que fueron entrenados, y tanto el ajuste fino
como el entrenamiento del generador lo leen de ahí en lugar de sus propios valores por
defecto. Una discrepancia falla al cargar los pesos con un error de forma.

#### 1.2.4 Fusión entre capas (MoLA)

La fusión no combina únicamente la salida final de cada rama. Cada capa de cada modalidad
aporta un token; todos se apilan y se someten a *self-attention* con 8 cabezas, y el
resultado se reduce con una suma ponderada por pesos aprendidos, uno por token. La
operación es *self-attention* en sentido estricto: el conjunto de tokens se atiende contra
sí mismo, con consulta, clave y valor idénticos. En el código de MoLA el módulo se llama
`cross_attention` porque cruza modalidades y profundidades, pero formalmente es
auto-atención sobre el conjunto apilado.

```
capa 1    [ token_grafo₁ , token_smiles₁ ]  ┐
capa 2    [ token_grafo₂ , token_smiles₂ ]  ├─ self-attention ── suma ponderada
capa 3    [ token_grafo₃ , token_smiles₃ ]  ┘   (8 cabezas)      (pesos aprendidos)
```

Con `use_graph` o `use_smiles` desactivados, el número de tokens por capa baja de 2 a 1 y el
vector de pesos se dimensiona en consecuencia: una rama apagada no aporta parámetros, en
lugar de aportar ceros que se seguirían contando y entrenando. Este detalle es el que hace
que la ablación de §3.4 compare modelos y no modelos con lastre.

La premisa de esta fusión no es solo que capas distintas codifiquen información distinta,
sino que **el objetivo de entrenamiento las moldea** para que así sea. Esa segunda mitad de
la premisa exige que las ramas se entrenen conjuntamente con la fusión, y es la razón por la
que el preentrenamiento descrito en §3.1 no congela nada.

### 1.3 Formulación de la tarea generativa

La mitad generativa del trabajo emplea el mismo encoder para escribir moléculas en vez de
leerlas. La tarea es `p(M_b | M_a, Δpropiedad)`: dada una molécula de partida y un
desplazamiento objetivo, producir una molécula análoga que lo satisfaga. Se trata, pues, de
generación condicionada en la que una de las condiciones es una molécula; en la literatura
de diseño de fármacos corresponde a optimización de *leads*.

Esta formulación no es una elección de conveniencia, sino la corrección de un defecto
concreto del diseño previo. Un generador condicionado que se entrena sobre
`(M → M, y = f(M))` tiene la molécula objetivo dentro del propio *memory* del decoder.
Reproducirla no requiere leer el token de condición: `H(objetivo | memory) = 0` antes
siquiera de mirarlo, y el gradiente no tiene motivo para enseñar al decoder a usarlo. El
condicionamiento resultante es nominal.

La solución consiste en que la molécula objetivo **no** sea la de entrada:

```
entrada al encoder   M_a   Clc1ccccc1   clorobenceno    logP 2,84
condición            Δ     −0,57
objetivo del decoder M_b   Fc1ccccc1    fluorobenceno   logP 2,27
```

Puesto que M_b no está en el *memory*, la única indicación de que hay que sustituir el cloro
por flúor, y no por bromo, es el Δ, con lo que el gradiente queda obligado a usarlo.

Este planteamiento elimina además el desajuste entre entrenamiento e inferencia. Con
condicionamiento absoluto se pedía `y = y_real ± 1σ`, una combinación que jamás aparece en
los datos; con un Δ relativo, en cambio, pedir `Δ = −1` en inferencia es exactamente lo que
el modelo vio miles de veces. La construcción del corpus de pares que materializa esta
formulación se detalla en §2.3.

### 1.4 Discretización de la condición

Cada uno de los cuatro Δ se discretiza en **20 bins por cuantiles**, más un bin nulo, y cada
bin tiene un *embedding* aprendido. Los bordes se ajustan **solo con los pares de
entrenamiento**.

La discretización sustituye a la proyección de un escalar crudo. Un número aislado a través
de un `Linear(1→H)` produce una señal de baja frecuencia y gradiente pobre; en el diseño
previo, además, entraba sin estandarizar, en kcal/mol. Un índice de bin, en cambio,
selecciona un vector de rango completo. Conviene precisar que bins y *features* de Fourier
resuelven el mismo problema, por lo que son alternativas y no se acumulan.

Los cuantiles se colapsan cuando la distribución es picuda —como ocurre con los Δ acumulados
en cero—; los bordes duplicados se funden para no crear bins inalcanzables cuyas filas del
embedding se entrenarían con cero ejemplos.

### 1.5 Composición del memory

```
memory = [ Δlogp │ Δtpsa │ Δqed │ Δmw │ átomo₁…átomoₙ │ char₁…char_L ]
           └────── condición ──────┘   └──── M_a, lo que hay que modificar ────┘
```

Los estados del encoder entran sin poolear: un vector por átomo y uno por carácter, no un
único vector resumen. El decoder hace *cross-attention* sobre toda la secuencia.

Se conserva el token prependido en lugar de recurrir a FiLM o AdaLN, porque el token se
ignoraba en el diseño previo por ser **redundante**, no por ser un token; eliminada la
redundancia, el mecanismo más simple debería bastar. Los diagnósticos de §3.5 determinan si
es así, y FiLM queda como alternativa documentada si no lo es.

### 1.6 Decoder y función de pérdida

El decoder es un `nn.TransformerDecoder` estándar: *self-attention* causal sobre los tokens
ya emitidos, *cross-attention* sobre el *memory* con máscara de acolchado, y pérdida de
entropía cruzada por token sobre los SELFIES de M_b, con *teacher forcing*.

El encoder conserva exactamente la configuración con la que fue preentrenado —`hidden = 256`
y 3 capas—, y no podría ser de otro modo: unos pesos preentrenados solo encajan en la forma
en que se entrenaron. El ancho del decoder queda fijado por el mismo valor, porque atiende,
vía *cross-attention*, sobre el *memory* que produce el encoder. En la implementación, los
scripts de fine-tuning y de generación leen esas dimensiones del propio checkpoint en lugar
de tomarlas de sus valores por defecto, de modo que la cadena permanece consistente aunque
el preentrenamiento se repita con otro tamaño.

El gradiente retrocede por la *cross-attention*, atraviesa el *memory* y alcanza tanto los
embeddings de condición como el encoder. El encoder queda así entrenado también por la señal
generativa, que es lo que permite que desactivar una de sus ramas degrade la generación de
forma medible.

---

## 2. Dataset y Preprocesamiento

### 2.1 Corpus de preentrenamiento

Se emplea **MOSES** (Molecular Sets), un subconjunto curado de ZINC restringido a espacio
químico *drug-like*. Se elige frente a volcados mayores por estar depurado y por disponer de
líneas base publicadas de validez, unicidad, novedad y FCD, lo que permite situar los
resultados generativos frente a la literatura.

#### 2.1.1 Canonicalización y codificación

Cada SMILES atraviesa cuatro pasos en RDKit: se parsea a molécula, se recanonicaliza
(`Chem.MolToSmiles`), se calcula su InChIKey como identificador estructural y se convierte a
**SELFIES** mediante `selfies.encoder`. Las moléculas que RDKit rechaza, o que SELFIES no
puede representar, se descartan.

La elección de SELFIES como espacio de salida del decoder es deliberada: `selfies.decoder`
no puede producir un SMILES inválido a partir de una cadena bien formada, de modo que la
validez química queda garantizada por construcción y no depende de que el modelo la aprenda.

#### 2.1.2 Vocabulario

El alfabeto SELFIES resultante consta de **29 tokens** (25 de SELFIES más `<PAD>`,
`<START>`, `<END>`, `<OTHER>`). Esa compacidad no es un error: MOSES está filtrado a un
espacio químico estrecho y su alfabeto SELFIES es genuinamente pequeño.

> **Garantía · validez por construcción.** No se aplica poda de tokens por frecuencia.
> Colapsar tokens raros en `<OTHER>` es la práctica habitual en modelado de lenguaje, pero
> aquí rompería la garantía de validez: `<OTHER>` no es decodificable, y una secuencia
> generada que lo contenga no es una molécula. Si se fija un umbral de frecuencia, se
> descartan las moléculas afectadas en lugar de corromperlas, de modo que todo identificador
> del vocabulario corresponde a SELFIES real.

Las secuencias se truncan al percentil 99,5 de longitud, **45 tokens** (mediana 35), y se
almacenan como una matriz `int16`. `<START>` y `<END>` se añaden en tiempo de entrenamiento
y no se almacenan.

#### 2.1.3 Control de fuga hacia los conjuntos de evaluación

Toda molécula del corpus cuyo InChIKey coincida con una de ESOL, FreeSolv o Lipophilicity se
elimina. Se usa InChIKey y no SMILES crudo para que una escritura distinta de la misma
molécula también se detecte.

La exclusión cubre los **tres splits** de los tres conjuntos, no solo el de test. Restringirla
al test bastaría para la evaluación, pero ataría el corpus a una partición concreta: al
cambiar la semilla o el criterio de partición, moléculas que antes estaban en entrenamiento
pasarían a test y el corpus habría que reconstruirlo. Cubriendo los tres splits, la partición
y el corpus dejan de depender el uno del otro.

| Conjunto | Filas | InChIKeys únicos | Eliminadas del corpus |
|---|---:|---:|---:|
| ESOL | 1 128 | 1 117 | 37 |
| FreeSolv | 642 | 642 | 0 |
| Lipophilicity | 4 200 | 4 200 | 169 |
| **Total** | **5 970** | **5 564** | **196** |

El total de claves únicas no es la suma de las tres filas: 5 959 − 5 564 = 395 moléculas
figuran en más de un dataset, y basta una coincidencia para excluirlas del corpus. La columna
de eliminadas sí es aditiva. La diferencia entre filas y claves únicas en ESOL son las once
duplicadas que §2.4 conserva deliberadamente; para esta exclusión da igual, porque lo que se
compara son conjuntos de claves.

El solape es del **3,5 %** de las 5 564 moléculas de evaluación. MOSES y estos tres conjuntos
son poblaciones casi disjuntas, de modo que la fuga entre las dos mitades del trabajo era
pequeña incluso antes de este filtro; eliminarla, aun así, cuesta 196 moléculas de 1,94 M.

El patrón es químicamente coherente: FreeSolv son disolventes pequeños y no se solapa con un
corpus drug-like, mientras que Lipophilicity procede de ChEMBL y aporta la mayoría de las
coincidencias.

El solape de *scaffolds*, en cambio, se mide y se reporta, pero **no** se elimina: 728
esqueletos de Murcko compartidos, que cubren 304 951 moléculas del corpus (**15,7 %**).
Eliminar todo scaffold compartido arrancaría quimiotipos completos de un corpus drug-like, y
ningún protocolo publicado de preentrenamiento lo hace; el número queda constando para que
el lector juzgue su alcance.

Tras estos filtros el corpus queda en **1 936 381 moléculas**.

### 2.2 Etiquetado con oráculo exacto

Depurado el corpus, falta dotarlo de señal supervisada. Para cada molécula se calculan
cuatro descriptores con RDKit:

| Propiedad | Función | Interpretación |
|---|---|---|
| logP | `Crippen.MolLogP` | coeficiente de reparto octanol/agua |
| TPSA | `rdMolDescriptors.CalcTPSA` | superficie polar topológica (Å²) |
| QED | `QED.qed` | estimación cuantitativa de *drug-likeness*, en [0, 1] |
| MW | `Descriptors.MolWt` | peso molecular (Da) |

Estos descriptores constituyen un **oráculo**, no un modelo: son funciones deterministas de
la estructura. De ahí se derivan tres propiedades metodológicas que el diseño explota:

- **No hay ruido de etiqueta**, a diferencia de una medida experimental.
- **No hay techo impuesto por un maestro**, a diferencia de pseudo-etiquetar con un predictor
  entrenado.
- **La evaluación no es circular**: en la generación condicionada se pide un objetivo y se
  mide lo obtenido con la misma función, sin ningún predictor aprendido en el bucle.

La alineación con las tareas finales tampoco es accidental. El logP de Crippen es el término
dominante de la ecuación de Delaney para solubilidad acuosa (ESOL), y el logD 7.4 de
Lipophilicity es logP corregido por ionización. El preentrenamiento no resuelve, por tanto,
una tarea auxiliar arbitraria, sino una versión exacta y abundante de la tarea de destino.

#### 2.2.1 Manejo de valores faltantes

Las moléculas cuyo cálculo falla se conservan con etiqueta `NaN` en lugar de eliminarse,
para que la fila *i* de la matriz de etiquetas siga correspondiendo a la fila *i* del corpus;
se filtran aguas abajo.

### 2.3 Minado de pares análogos

Esta etapa construye la señal de entrenamiento de la mitad generativa y materializa la
formulación `p(M_b | M_a, Δ)` introducida en §1.3.

#### 2.3.1 Procedimiento

1. **Agrupación por scaffold de Murcko.** Comparar 1,94 M × 1,94 M moléculas es inviable.
   Los análogos se concentran, casi por definición, entre moléculas que comparten esqueleto,
   de modo que agrupar convierte un problema cuadrático en uno lineal. El scaffold es
   únicamente un mecanismo de búsqueda: **nunca entra en la red**.
2. **Muestreo de candidatos.** Dentro de cada grupo se comparan hasta 200 candidatos por
   molécula en lugar del grupo entero, lo que acota el coste con independencia de lo poblado
   que esté un scaffold. Los grupos de más de 5 000 miembros se parten en bloques para
   equilibrar la carga.
3. **Banda de similitud.** Se aceptan pares con Tanimoto sobre huellas ECFP4 en
   `[0,50 , 0,95]`. Por debajo de 0,50 no hay una modificación sino otra molécula, y el
   encoder no puede ayudar; por encima de 0,95 hay una copia, que no enseña nada.
4. **Preferencia por el análogo más lejano.** Cuando hay más candidatos admisibles que el
   tope de 10 por molécula, se retienen los *menos* similares: son los que conllevan el mayor
   cambio estructural y, por tanto, el mayor Δ que la condición debe explicar.
5. **Ambas direcciones.** De cada par hallado se emiten `(a, b)` y `(b, a)`. El Δ es una
   magnitud con signo, y entrenar en una sola dirección sesgaría cada token de condición
   hacia un signo.
6. **Pares identidad.** Un 5 % de los ejemplos son `(M, M)` con Δ = 0. Fijan la semántica de
   «no cambies nada» y proporcionan una prueba de sanidad en evaluación: pedir Δ = 0 debe
   devolver la semilla.

#### 2.3.2 Estadística de los pares obtenidos

El resultado son **12 991 314 pares**, de los cuales 649 566 son de identidad.

| Propiedad | desv. típica de Δ | p1 | p99 | \|Δ\| > 1 σ |
|---|---:|---:|---:|---:|
| logP | 0,685 | −1,766 | +1,766 | 27,7 % |
| TPSA | 15,784 | −43,09 | +43,09 | 29,4 % |
| QED | 0,066 | −0,198 | +0,198 | 21,9 % |
| MW | 27,679 | −69,09 | +69,09 | 32,1 % |

La media es exactamente cero y los percentiles son simétricos, lo que confirma que la emisión
bidireccional funciona. La última columna usa unidades de desviación típica porque un umbral
absoluto carece de sentido comparativo: «más de 1» es trivial para MW en daltons e
inalcanzable para QED, cuyo rango completo es [0, 1].

El percentil 99 del Δ de logP, **±1,77**, define el rango de control que la evaluación puede
solicitar. Pedir desplazamientos mayores sería extrapolación, precisamente el defecto que
este diseño corrige.

### 2.4 Conjuntos de regresión experimental

Frente al oráculo exacto del corpus, la primera tarea confronta lo aprendido con medidas
experimentales. Se emplean tres conjuntos de regresión de MoleculeNet:

| Conjunto | Propiedad | Moléculas | Test |
|---|---|---:|---:|
| ESOL (Delaney) | solubilidad acuosa, log mol/L | 1 128 | 113 |
| FreeSolv (SAMPL) | energía libre de hidratación, kcal/mol | 642 | 65 |
| Lipophilicity | logD a pH 7,4 | 4 200 | 420 |

Los CSV se descargan del repositorio público de MoleculeNet, se canonicalizan y se parten con
un **split aleatorio** (80/10/10).

**Los duplicados no se fusionan.** MoleculeNet distribuye ESOL con once moléculas repetidas
por InChIKey —el mismo compuesto escrito de dos formas, y en algún caso con medidas que
discrepan: el sorbitol aparece con 0,060 y con 1,090—. Fusionarlas promediando produce un
conjunto más limpio y, a la vez, un conjunto **distinto**: 1 117 filas donde toda la
literatura midió sobre 1 128. El RMSE deja entonces de ser comparable con las cifras
publicadas y con cualquier otro grupo que corra el mismo benchmark. Entre un efecto pequeño
de memorización y unos números incomparables, se conserva el benchmark tal como se
distribuye y el efecto se reporta en lugar de eliminarse.

> **Cuantificación del efecto conservado.** Con un reparto 80/10/10, la probabilidad de que
> un par duplicado quede repartido entre entrenamiento y test es 2·0,8·0,1 = 0,16, de modo
> que se esperan ~1,8 de los once pares, es decir un ~1,6 % del conjunto de test de ESOL. La
> discrepancia mediana entre copias es 0,000: casi todas coinciden, y solo el sorbitol se
> contradice de verdad. La deduplicación **sí** se mantiene, en cambio, entre el corpus de
> preentrenamiento y los conjuntos de evaluación (§2.1.3), que es una cuestión distinta: allí
> no se modifica ningún benchmark, se impide que el encoder se entrene sobre las moléculas
> con las que después se le evalúa.

La elección de partición aleatoria, y no por scaffold, sigue la recomendación del propio
artículo de MoleculeNet para estos tres conjuntos: son tareas de fisicoquímica, donde la
propiedad depende de la composición atómica y de descriptores globales más que del esqueleto,
y es el protocolo bajo el que se publicaron las cifras con las que estos resultados se
comparan. La partición por scaffold es la convención de MoleculeNet para sus conjuntos de
clasificación biológica, donde el esqueleto sí determina la actividad. Usar scaffold aquí
daría números más bajos que no serían comparables con la literatura y que medirían una
dificultad distinta de la que el conjunto plantea.

Cada partición se acompaña de un `split_meta.json` que registra el criterio, la semilla, los
tamaños y si se fusionaron duplicados, porque esa última decisión determina de qué benchmark
se trata y dos conjuntos de resultados que difieran en ella no son promediables.

> **Nota · dos particiones distintas en un mismo trabajo.** La partición aleatoria de §2.4 y
> la partición por scaffold de los pares (§2.6) responden a preguntas distintas y conviven sin
> contradicción. En §2.4 se mide la capacidad de predecir una propiedad fisicoquímica y la
> referencia publicada es aleatoria. En §2.6 lo que se mide es la generación de un análogo, y
> ahí el scaffold es exactamente lo que el modelo podría memorizar: un par cuyo origen esté en
> entrenamiento y cuyo destino esté en test mediría memorización, no generalización.

### 2.5 Cobertura de distribución entre el corpus y los conjuntos de evaluación

Una transferencia solo puede apoyarse en la parte del espacio químico que el corpus
realmente visita. Medido sobre muestras de 20 000 moléculas, en átomos pesados (percentiles
5 / 50 / 95) y peso molecular mediano:

| Conjunto | p5 | p50 | p95 | MW mediana |
|---|---:|---:|---:|---:|
| MOSES | 17 | 21 | 25 | 300 |
| ZINC-250k | 18 | 22 | 25 | 315 |
| FreeSolv | 4 | 8 | 18 | 120 |
| ESOL | 4 | 12 | 25 | 184 |
| Lipophilicity | 15 | 27 | 38 | 388 |

MOSES es *ZINC Clean Leads* filtrado a un peso molecular de 250-350, de modo que ocupa una
banda de 17 a 25 átomos pesados. La consecuencia, que no era evidente antes de medirla, es
que el corpus **no cubre bien ninguno de los tres conjuntos**, y falla por los dos extremos:
FreeSolv queda íntegramente por debajo, la mediana de ESOL cae por debajo del percentil 5 del
corpus, y más de la mitad de Lipophilicity supera su percentil 95. Que Lipophilicity sea un
conjunto *drug-like* no lo hace del tamaño de MOSES.

La misma medición descarta la opción aparentemente obvia de escalar el corpus con más ZINC:
ZINC-250k ocupa esa banda con los mismos percentiles, así que diez millones de moléculas
adicionales añadirían volumen y ninguna cobertura nueva.

Este resultado cualifica el hallazgo de §3.2 en lugar de contradecirlo: el preentrenamiento
mejora los tres conjuntos pese a una cobertura de distribución pobre, lo que hace la
transferencia más notable, no menos. Y fija la dirección de la única ampliación de corpus que
tiene sentido probar: hacia arriba y hacia abajo del rango, no hacia más volumen dentro de
él.

> **Un caso donde la exclusión de §2.1.3 dejó de ser rutinaria.** Al evaluar QM9 (129 440
> moléculas de ≤9 átomos pesados) como candidato para el extremo pequeño, la comprobación de
> solapamiento arrojó que QM9 contiene el **35,5 % de FreeSolv** (228 de 642) y el 21,7 % de
> ESOL. Concatenarlo sin deduplicar habría puesto un tercio de un conjunto de evaluación
> dentro del corpus de preentrenamiento, y el resultado de FreeSolv habría mejorado por
> memorización sin que nada lo delatara. La exclusión por InChIKey, que frente a MOSES
> eliminaba 196 moléculas de 1,94 M, aquí evita una fuga de esa magnitud a un coste del
> 0,25 % de QM9.

### 2.6 Partición de los pares

Los pares, por su parte, se parten **por scaffold**, no por par. Como el minado opera dentro
de grupos de scaffold, asignar esqueletos completos a una partición mantiene todos los pares
íntegros y garantiza que ninguna molécula aparezca en dos particiones.

> **Garantía · fuga entre particiones.** Una partición por par colocaría la misma molécula en
> entrenamiento y en test, como origen en una y como destino en la otra, y las cifras
> retenidas medirían memorización. Verificado sobre datos sintéticos minados con el mismo
> procedimiento: la partición por scaffold deja el 100 % de los pares íntegros y los conjuntos
> de moléculas disjuntos, mientras que una partición por par de los mismos datos filtraría
> decenas de miles de moléculas a ambos lados.

La partición de validación del preentrenamiento, en cambio, se detalla junto a su
configuración en §3.1, por ser un 1 % aleatorio sobre un corpus del que el test ya se ha
excluido en §2.1.3.

---

## 3. Estrategia de Entrenamiento

### 3.1 Preentrenamiento supervisado del encoder

El encoder se entrena sobre el corpus completo con una cabeza de regresión multitarea que
predice las cuatro etiquetas de RDKit simultáneamente. El interés no está en esos cuatro
valores, sino en la representación que el encoder desarrolla al producirlos.

#### 3.1.1 Estandarización de objetivos

Las cuatro propiedades se estandarizan con la media y la desviación del conjunto de
entrenamiento, medidas sobre el corpus completo:

| Propiedad | media | desv. típica |
|---|---:|---:|
| logP | 2,446 | 0,927 |
| TPSA | 65,797 | 18,089 |
| QED | 0,806 | 0,095 |
| MW | 307,361 | 27,995 |

Sin estandarizar, un error cuadrático medio sin ponderar estaría dominado por el peso
molecular —desviación típica de casi 28 daltons— frente al QED, cuya escala completa es el
intervalo [0, 1] y cuya desviación es de 0,095; el modelo resultante sería un regresor de MW
con tres adornos. El RMSE de validación se reporta de vuelta en las unidades propias de cada
propiedad.

Conviene no confundir esta tabla con la de §2.3.2: allí las desviaciones son las de los **Δ**
entre pares, aquí las de los **valores absolutos**. Que la de MW casi coincida en ambas
(27,679 frente a 27,995) es una coincidencia debida al estrecho rango de masas de MOSES, no
una repetición de la misma cifra.

#### 3.1.2 Configuración

| Parámetro | Valor | Justificación |
|---|---|---|
| Pasos | 40 000 | presupuesto fijo, idéntico entre configuraciones |
| Tamaño de lote | 256 | — |
| Optimizador | AdamW fusionado | — |
| Planificación | coseno tras 1 000 pasos de calentamiento | — |
| Precisión | bf16 (pérdida en fp32) | — |
| Partición de validación | 1 % | aleatoria; el test ya se excluyó del corpus |

Como todos los pesos son entrenables desde una inicialización aleatoria, se emplea una única
tasa de aprendizaje. Distinguir entre tasa del cuerpo y tasa de la cabeza solo tiene sentido
cuando hay un *backbone* preentrenado que proteger.

### 3.2 Fine-tuning para predicción de propiedades experimentales

Concluido el preentrenamiento, el encoder se transfiere a los tres conjuntos descritos en
§2.4, con la cabeza de regresión reinicializada a una única salida.

#### 3.2.1 El contraste que se mide

Cada configuración se entrena dos veces, idénticas salvo en el origen de los pesos:

- **Sin preentrenar** — inicialización aleatoria; el modelo solo ve las 650 a 4 200 moléculas
  del conjunto.
- **Preentrenada** — parte del checkpoint del preentrenamiento, descartando su cabeza.

La primera fila no es prescindible. «RMSE 0,85 con preentrenamiento» no significa nada por sí
solo: el resultado es la diferencia entre ambas.

> **Garantía · consistencia del vocabulario.** Al inicializar desde el checkpoint, el
> vocabulario de caracteres se toma del propio checkpoint y no se reconstruye a partir del
> conjunto pequeño. El *embedding* de SMILES está indexado por identificador de carácter:
> reconstruir el vocabulario haría que cada fila de esa matriz preentrenada pasara a
> representar un carácter distinto. El modelo entrenaría, convergería, y estaría mal, sin
> emitir ningún error.

#### 3.2.2 Métricas y protocolo de comparación

Se reportan RMSE (métrica principal, en las unidades del objetivo), MAE, R² y NRMSE, esta
última normalizada por la desviación típica del conjunto de entrenamiento —y no por el
rango— para que un único valor extremo en un test pequeño como el de FreeSolv no la
distorsione. Cada celda se repite con tres semillas (2025, 2026, 2027).

**La partición se rehace en cada semilla**, con el mismo particionador aleatorio de DeepChem
que emplea el trabajo de referencia, sobre el conjunto recombinado. Una partición congelada
responde «¿es este modelo mejor que aquel?», porque todas las filas ven exactamente los
mismos datos, pero no puede responder «¿es bueno este número»: un test de 113 moléculas es un
único sorteo, y repartir de nuevo los mismos datos mueve el RMSE de un modelo sin cambios de
0,485 a 0,663. Con la partición fija, la dispersión reportada cubre únicamente la
inicialización de pesos y oculta un término un orden de magnitud mayor.

**La comparación entre filas es pareada.** Las dos filas —desde cero y preentrenada— recorren
las mismas semillas, y al rehacer la partición por semilla eso significa que ven las mismas
particiones. Contrastar medias independientes descarta ese emparejamiento y carga al error
una varianza que se cancela exactamente al restar; como esa varianza es aproximadamente diez
veces el efecto, el contraste no pareado llega a declarar ruido una diferencia real.

**Y se reporta el signo además de la magnitud.** El beneficio del preentrenamiento depende de
qué moléculas caen en test —un test fácil estrecha la diferencia, uno difícil la ensancha—,
de modo que la dispersión pareada arrastra una interacción real entre partición y tratamiento
y no solo ruido. El número de semillas que apuntan en la misma dirección no sufre ese
problema, y es la afirmación más robusta cuando la magnitud varía.

> **Nota · comparabilidad del NRMSE.** El NRMSE definido aquí (RMSE / σ del entrenamiento) no
> es el de la implementación de referencia, que normaliza por el rango. Los RMSE sí son
> directamente comparables; los NRMSE no. Del mismo modo, las desviaciones publicadas por la
> referencia usan `ddof=0` y las de aquí `ddof=1`: con tres semillas el factor es √(3/2) =
> 1,22, suficiente para cambiar si una diferencia cae dentro o fuera de la dispersión.

### 3.3 Entrenamiento del decoder

El decoder descrito en §1.6 tiene 6 capas y se entrena 60 000 pasos con lote 256.

#### 3.3.1 Dropout de condición y guía sin clasificador

Durante el entrenamiento, cada propiedad se sustituye por su bin nulo con probabilidad 0,15,
**de forma independiente** entre propiedades. Esa independencia permite que una petición
nombre una propiedad y deje las demás sin especificar —«Δ logP = +1, el resto me da igual»—,
que es la petición realista; un *dropout* conjunto solo permitiría condicionamiento de todo o
nada.

Eso habilita, a su vez, *classifier-free guidance* en muestreo:

```
logits = logits_sin_condición + w · (logits_con_condición − logits_sin_condición)
```

El peso `w` regula la intensidad del control y permite un barrido completo de resultados con
un solo modelo entrenado. Es además un diagnóstico sin coste adicional de entrenamiento:
**si aumentar `w` no cambia nada, la condición no se está utilizando.**

### 3.4 Ablación y presupuesto de cómputo

Tres brazos, idénticos salvo en las ramas activas del encoder: **grafo + SMILES**, **solo
grafo**, **solo SMILES**. Se ejecutan en las dos mitades del trabajo, de acuerdo con el eje
único anunciado en §1.1.

El presupuesto se fija en **pasos**, no en épocas, tanto en el preentrenamiento (§3.1.2) como
en el decoder (§3.3). Los brazos se ejecutan secuencialmente sobre una única GPU, y un brazo
que entrenara más tiempo por tener épocas más baratas convertiría la comparación en una sobre
tiempo de reloj en lugar de sobre las modalidades.

Por el mismo motivo las dos ramas se equiparan en parámetros (§1.2.3): igualar el presupuesto
de pasos y dejar que una rama tenga 8,75× los parámetros de la otra sería controlar una
variable de confusión y no la otra.

A estos tres brazos se añade una cuarta corrida que repite **grafo + SMILES** partiendo del
encoder preentrenado sobre MOSES. No es un cuarto brazo de modalidad, sino el segundo eje del
diseño: los brazos 1 a 3 responden *¿aporta la fusión multimodal?* y la comparación entre el
brazo 1 y esta cuarta corrida responde *¿aporta el preentrenamiento?*. El brazo fusionado es
el pivote de ambos ejes, y por eso son cuatro corridas y no seis.

### 3.5 Diagnósticos de uso de la condición

El fallo del diseño previo, tal como se expuso en §1.3, no fue que el condicionamiento
funcionara mal, sino que no existía sin que nada lo delatara: la pérdida bajaba, las
moléculas eran válidas, y el token simplemente no se leía. Por eso el método incorpora tres
comprobaciones cuyo único propósito es detectar esa situación, independientes de la calidad
de lo generado.

**Barrido del peso de guía.** Se muestrea el mismo conjunto de semillas y condiciones con `w`
creciente. Si el control de propiedad no mejora al aumentar `w`, la condición no está
interviniendo en la predicción. No requiere reentrenar y su coste es una pasada adicional por
muestreo.

**ΔNLL con la condición alterada.** Se recalcula la entropía cruzada de validación
sustituyendo los tokens de condición por variantes que no llevan información sobre el par:
bin nulo, bins aleatorios, y —la prueba decisiva— los bins **barajados dentro del lote**, que
conservan exactamente la distribución marginal y solo rompen el emparejamiento entre molécula
y Δ. Se reporta la diferencia respecto de la condición correcta con un intervalo de confianza
*bootstrap* emparejado por molécula. Si ese intervalo contiene el cero, la condición no aporta
información incremental y el condicionamiento es nominal. El intervalo es imprescindible: sin
él, un ΔNLL pequeño no se distingue del ruido.

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

### 3.6 Evaluación generativa con oráculo

El protocolo de evaluación elimina la circularidad de raíz:

1. Se toman moléculas semilla del **split de test**, es decir, pertenecientes a scaffolds que
   el modelo no vio en ningún par de entrenamiento. Por la partición descrita en §2.6,
   ninguna de esas moléculas apareció como origen ni como destino.
2. Se solicita un Δ dentro del rango que los datos soportan, derivado de los percentiles de la
   tabla de §2.3.2.
3. Se muestrea y se decodifica SELFIES a SMILES.
4. **Se mide la propiedad real de lo generado con RDKit** y se compara con lo solicitado.

El contraste con la evaluación anterior es directo: aquella puntuaba las moléculas generadas
con la cabeza de regresión del propio checkpoint generativo —encoder compartido y misma
pérdida, es decir, el modelo evaluándose a sí mismo—, mientras que RDKit no comparte nada con
el modelo.

Las métricas reportadas son: tasa de acierto (fracción de generadas cuyo Δ real cae en el bin
solicitado), MAE entre Δ pedido y obtenido, curva pedido-vs-obtenido —cuya pendiente es la
fuerza del condicionamiento—, validez, unicidad, novedad, FCD, similitud de Tanimoto a la
semilla (alta pero inferior a 1; si es 1, el modelo copia), y barrido del peso de guía.

---

---

## 4. Resultados

Todas las cifras de esta sección proceden de corridas concretas y cada bloque indica sobre
qué estado del pipeline se midió, porque el protocolo evolucionó durante la experimentación y
dos bloques medidos bajo protocolos distintos no son promediables.

### 4.1 Coste computacional

El preentrenamiento del encoder sobre 1 936 381 moléculas consume **11 min 39 s** en una RTX
PRO 6000 Blackwell: 40 000 pasos a 57,2 pasos/s, es decir 14 649 moléculas por segundo y
10 240 000 ejemplos vistos, unas 5,3 pasadas sobre el corpus. Cada brazo del decoder emplea
entre 16 y 31 minutos para el mismo presupuesto de 60 000 pasos.

La consecuencia de diseño es que **el cómputo no limita el tamaño del corpus**. Con el
presupuesto fijado en pasos, ampliar el corpus no alarga el entrenamiento en absoluto; solo
cambia cuántas veces se ve cada molécula. Manteniendo constantes las pasadas, decuplicar el
corpus costaría menos de dos horas. El límite real es la memoria: la caché de grafos ocupa
1,4 GB por cada 1,94 M de moléculas y se carga entera en RAM.

### 4.2 Mitad predictiva: transferencia desde MOSES

Medido con repartición por semilla (particionador aleatorio de DeepChem), tres semillas,
contraste pareado. Ambas filas comparten arquitectura HybridMoLA y difieren únicamente en el
origen de los pesos iniciales.

| Conjunto | | RMSE | MAE | R² | Δ pareada | semillas |
|---|---|---:|---:|---:|---:|:---:|
| ESOL | desde cero | 0,775 ± 0,073 | 0,572 | 0,853 | | |
| | preentrenado | **0,640 ± 0,014** | 0,480 | 0,901 | −0,134 ± 0,061 | 3/3 |
| FreeSolv | desde cero | 1,308 ± 0,041 | 0,970 | 0,859 | | |
| | preentrenado | **0,903 ± 0,154** | 0,547 | 0,934 | −0,406 ± 0,194 | 3/3 |
| Lipophilicity | desde cero | 0,849 ± 0,018 | 0,656 | 0,508 | | |
| | preentrenado | **0,675 ± 0,019** | 0,519 | 0,689 | −0,174 ± 0,029 | 3/3 |

**El preentrenamiento sobre MOSES mejora los tres conjuntos, con las nueve semillas en la
misma dirección.** Bajo la hipótesis nula de ausencia de efecto, nueve de nueve sobre
particiones independientes tiene probabilidad 0,5⁹ = 0,002. Enunciado por conjunto por
separado no alcanzaría significación —tres de tres da p = 0,125—, de modo que la afirmación se
formula sobre las tres tareas conjuntamente.

El efecto es mayor en FreeSolv (−0,406) que en los otros dos, lo que resulta contrario a lo
que la cobertura de distribución (§2.5) haría esperar: FreeSolv es el conjunto que MOSES peor
cubre. La mejora en R² de Lipophilicity, de 0,508 a 0,689, es la más pronunciada en términos
de varianza explicada y corresponde al conjunto más grande, donde la dispersión entre semillas
es menor.

> **Sobre la reproducibilidad de estas cifras.** Una medición anterior de la misma tabla, con
> las mismas semillas y el mismo particionador, arrojó valores distintos: 0,611 / 1,139 / 0,671
> en la fila preentrenada. La diferencia no procede de indeterminismo numérico sino del orden
> de las filas del conjunto. La variante que fusionaba duplicados agrupaba por InChIKey, y esa
> operación **reordena** el conjunto aunque no fusione ninguna fila; al conservarlos, el orden
> pasa a ser el del fichero original. El particionador permuta índices, de modo que la misma
> permutación sobre un orden distinto selecciona moléculas distintas. Ambas mediciones son
> válidas y corresponden a dos particiones legítimas; las tres semillas favorecen al modelo
> preentrenado en los tres conjuntos bajo las dos, lo que suma dieciocho comparaciones
> concordantes sobre dos ordenaciones independientes.
>
> El entrenamiento se ejecuta además con determinismo de cuDNN desactivado, de modo que dos
> corridas idénticas pueden diferir en la tercera o cuarta cifra decimal. Ese término es un
> orden de magnitud menor que el anterior y no explica diferencias como las observadas aquí.

### 4.3 Mitad generativa: el condicionamiento funciona

Evaluación con oráculo RDKit (§3.6): se solicita un Δ logP, se genera, y se mide con RDKit el
Δ realmente obtenido. Cada bin se pide a las **mismas** 100 moléculas semilla del split de
test, de modo que la correlación es intra-semilla y no puede originarse en qué moléculas
tocaron. Ningún predictor aprendido interviene.

| Brazo | w | ρ | pendiente | validez | unicidad | novedad | copia | Tanimoto |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| grafo + SMILES, desde cero | 1 | +0,811 | +0,900 | 1,000 | 0,930 | 0,854 | 0,040 | 0,520 |
| | 3 | +0,856 | +1,145 | 1,000 | 0,950 | 0,874 | 0,038 | 0,490 |
| grafo + SMILES, preentrenado | 1 | +0,778 | +0,893 | 1,000 | 0,958 | 0,885 | 0,025 | 0,463 |
| | 3 | +0,806 | +1,158 | 1,000 | 0,951 | 0,904 | 0,014 | 0,435 |
| solo grafo | 1 | +0,786 | +0,866 | 1,000 | 0,935 | 0,857 | 0,039 | 0,522 |
| | 3 | +0,851 | +1,114 | 1,000 | 0,959 | 0,877 | 0,035 | 0,482 |
| solo SMILES | 1 | +0,812 | +0,885 | 1,000 | 0,933 | 0,845 | 0,044 | 0,525 |
| | 3 | +0,871 | +1,131 | 1,000 | 0,950 | 0,876 | 0,034 | 0,481 |

**El generador obedece.** ρ entre 0,78 y 0,87 sobre los veinte bins contiguos, con validez
perfecta —garantizada por SELFIES—, novedad entre 0,85 y 0,90 y una tasa de copia por debajo
del 4,4 %. Esa última columna es la que protege a ρ: un modelo que devolviera la semilla sin
cambios tendría Δ constante y puntuaría bien en los bins centrales sin haber aprendido nada.

**El control nulo confirma de dónde viene la señal.** Con la condición puesta al bin nulo,
sobre las mismas semillas, el Δ se reparte alrededor de −0,15 a −0,26 con desviación de 0,64 a
0,78. Condicionar desplaza la media a lo largo de ±1,3 y estrecha la dispersión a 0,3-0,5 por
bin. Si el brazo nulo se pareciera a los bins solicitados, ρ provendría de otra fuente.

**La guía sin clasificador actúa.** ρ aumenta en los cuatro brazos al pasar de w = 1 a w = 3, y
la pendiente pasa de ~0,89 a ~1,14. Era el diagnóstico gratuito que ese mecanismo compra: si la
condición se ignorase, w no cambiaría nada. Que a w = 3 la pendiente exceda 1,0 es
sobrecorrección, el comportamiento esperado.

**El Tanimoto a la semilla desciende al aumentar w** (0,520 → 0,490 en el brazo fusionado), de
modo que apretar el control aleja del compuesto de partida. Es un compromiso real entre control
y similitud, y un resultado por sí mismo.

### 4.4 Ablación de modalidades y de preentrenamiento: resultado nulo

Los cuatro brazos difieren en 0,033 de ρ a w = 1 y 0,065 a w = 3. Con una única semilla de
entrenamiento por brazo, eso no es una diferencia defendible.

**Ni la fusión multimodal ni el preentrenamiento aportan en la mitad generativa.** El brazo
fusionado no supera a ninguna de las dos modalidades por separado, y el inicializado desde el
encoder preentrenado en MOSES queda último en ambos pesos de guía, con un margen que se ensancha
a w = 3 (0,806 frente a 0,851–0,871).

Las pérdidas de entrenamiento concuerdan: el brazo preentrenado alcanza 0,1249 frente al 0,0953
del mismo brazo desde cero. Es peor en reconstrucción **y** en condicionamiento.

| Brazo | Parámetros | Pérdida val. | Precisión de token | Minutos |
|---|---:|---:|---:|---:|
| grafo + SMILES, desde cero | 14 113 319 | 0,0953 | 0,9628 | 31 |
| grafo + SMILES, preentrenado | 14 113 319 | 0,1249 | 0,9510 | 31 |
| solo grafo | 10 135 588 | 0,1047 | 0,9588 | 16 |
| solo SMILES | 10 893 857 | 0,0981 | 0,9616 | 26 |

Esa pérdida de validación mide reconstrucción de SELFIES, no condicionamiento: como los pares
minados tienen Tanimoto ≥ 0,50, un modelo que copiase la semilla puntuaría bien sin haber
aprendido a condicionar. Se reporta para constatar que los cuatro entrenamientos convergieron,
no para compararlos.

### 4.5 El contraste entre las dos mitades

El preentrenamiento sobre MOSES **mejora la predicción de propiedades y no mejora la generación
condicionada**; en la segunda, apunta a perjudicarla.

Las columnas del oráculo respaldan un mecanismo desde tres direcciones independientes. El brazo
preentrenado exhibe menor Tanimoto a la semilla (0,435–0,463 frente a ~0,49–0,52), menor tasa de
copia (0,014–0,025 frente a ~0,04) y un control nulo más negativo (−0,222). Se aleja más de la
molécula de entrada que los demás: **la usa menos**. Y es precisamente el de peor ρ.

Eso concuerda con el objetivo bajo el que se preentrenó. La cabeza de regresión premia que la
propiedad sea linealmente legible desde el vector pooleado, que es exactamente lo que la mitad
predictiva necesita —y sus resultados lo confirman— pero no lo que el decoder necesita, que es
detalle estructural para reconstruir un análogo. El preentrenamiento parece haber descartado
justo eso.

La afirmación se sostiene sobre una sola semilla de entrenamiento por brazo y se enuncia, por
tanto, como hipótesis explicativa consistente con cuatro observaciones independientes, no como
mecanismo demostrado.

---

## Limitaciones

- **Desplazamiento de dominio en el preentrenamiento, cuantificado y sin cerrar.** MOSES
  abarca de 17 a 25 átomos pesados y falla por los dos extremos: FreeSolv queda entero por
  debajo, la mitad de ESOL por debajo y más de la mitad de Lipophilicity por encima (§2.5). La
  transferencia medida en §4.2 ocurre pese a esa cobertura, no gracias a ella. La ampliación
  del corpus hacia los dos extremos está identificada pero no ejecutada.
- **No hay generación *de novo*.** El encoder necesita una entrada, de modo que no es posible
  generar sin molécula de partida. Es la contrapartida directa de exigir que el encoder sea
  imprescindible: la generación *de novo* carece de encoder y, por tanto, de fusión multimodal
  que ablatar. En optimización de *leads*, que es el caso de uso al que corresponde la tarea,
  siempre existe una molécula de partida.
- **Rango de control acotado por los datos.** El percentil 99 del Δ de logP es ±1,77;
  solicitar desplazamientos mayores sería extrapolación.
- **Una sola semilla de entrenamiento en la mitad generativa.** Los cuatro brazos de §4.4 se
  entrenaron una vez cada uno. El resultado nulo de la ablación y la desventaja consistente del
  brazo preentrenado descansan, por tanto, sobre una única inicialización por brazo. Elevar la
  afirmación de observación a resultado exige repetir los cuatro con dos semillas más.

- **El contraste predictivo está medido sobre el benchmark con duplicados fusionados.** §2.4
  documenta la decisión de conservarlos para preservar la comparabilidad con la literatura, y
  §4.2 se midió antes de ese cambio. La repetición está pendiente.

- **El particionador compatible con DeepChem no está verificado.** §3.2.2 reproduce
  `dc.splits.RandomSplitter` sin importar DeepChem, a partir de su comportamiento documentado.
  La aritmética de cortes está confirmada —los tamaños de partición que produce coinciden con
  los esperados de esa implementación— pero no la permutación. Importa para situar estos
  resultados junto a los publicados, que se midieron con ese particionador. Hasta ejecutar
  `scripts/verify_splitter.py` contra DeepChem real, la formulación defendible es «mismo
  procedimiento de partición», no «particiones idénticas».

- **Dominancia de una modalidad: descartada como explicación, sin alternativa.** La ablación
  de §4.4 no muestra que una rama asuma el trabajo —los brazos de una sola modalidad igualan al
  fusionado— sino que la fusión no aporta nada medible sobre cualquiera de ellas. Por qué el
  segundo eje de información no ayuda queda sin explicar.

---

## Pendiente de completar

- **Repetir §4.2 sobre el benchmark sin fusionar duplicados**, para que las magnitudes
  correspondan al conjunto que §2.4 define.
- **Dos semillas más para los cuatro brazos generativos** (§4.4), unas seis horas de GPU, que
  convierten la desventaja del brazo preentrenado de observación en resultado.
- **Ejecutar `scripts/verify_splitter.py`** donde haya DeepChem instalado, para poder afirmar
  particiones idénticas en lugar de procedimiento idéntico.
- **Ampliar el corpus hacia los extremos identificados en §2.5**: QM9 para el régimen pequeño
  —evaluado, con el solapamiento con FreeSolv ya medido— y una fuente de moléculas mayores para
  Lipophilicity. Coste de cómputo despreciable (§4.1); lo que está por ver es si mejora.
- **Situar los resultados frente a la literatura publicada.** Al no reportarse ninguna
  reimplementación de referencia, esta es la única comparación externa del trabajo, y de ella
  depende que la mitad predictiva pueda leerse como competitiva y no solo como internamente
  consistente. Requiere anotar de cada trabajo citado su criterio de partición, su número de
  semillas, si fusionó duplicados y en qué unidades reporta el error.
- **Figuras**: la curva pedido-vs-obtenido por brazo, y la tabla pareada con el conteo de
  signos.

---
