"""Generación condicionada partiendo de moléculas ajenas al corpus de preentrenamiento.

    python -m crossmodal_model.generation.eval_fuera_corpus \
        --ckpt checkpoints/pairs/pairs_graph_smiles_pretrained_s2025.pt

Mide lo que la evaluación con oráculo no puede medir: si el control sobrevive fuera de la
banda de espacio químico que el corpus ocupa. Las moléculas de partida salen de ESOL,
FreeSolv y Lipophilicity, que según la medición de cobertura caen por debajo y por encima
de esa banda, y se eligen recorriendo el rango de tamaños de cada conjunto.

Nada del camino de generación se reimplementa: se importa de eval_oracle.py. Lo único
propio es de dónde salen las moléculas de partida y cómo se featurizan.

Dos comprobaciones que esta evaluación necesita y la del corpus no.

La primera es la pertenencia. Una molécula de un conjunto de evaluación puede estar
también en MOSES, y entonces no es ajena al corpus y no prueba nada sobre generalización.
Se descartan por identificador estructural antes de generar, y se informa de cuántas.

La segunda es el vocabulario de caracteres. La rama de SMILES se entrenó con el
vocabulario de MOSES, y un carácter que no esté en él se codifica como relleno, en
silencio. Sobre moléculas ajenas al corpus eso deja de ser improbable, de modo que se
cuenta y se informa: una degradación del control podría deberse a eso y no al mecanismo de
condicionamiento, y sin la cifra no habría forma de distinguirlo.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from crossmodal_model.generation.eval_oracle import (  # noqa: E402
    PROPERTIES, canonical_and_key, generate_for_request, load_generator, property_values,
)
from crossmodal_model.generation.pair_data import MoleculeGraphCache  # noqa: E402
from crossmodal_model.train.core import DATASETS  # noqa: E402

CONJUNTOS = {"esol": "ESOL", "freesolv": "FreeSolv", "lipo": "Lipophilicity"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--corpus-dir", default="data/moses")
    p.add_argument("--property", default="logp", choices=PROPERTIES)
    p.add_argument("--por-conjunto", type=int, default=30,
                   help="moléculas de partida de cada conjunto, repartidas por tamaño")
    p.add_argument("--bins", type=int, nargs="+", default=[0, 5, 10, 14, 19],
                   help="intervalos solicitados; los mismos que la rejilla por omisión")
    p.add_argument("--guidance", type=float, nargs="+", default=[1.0, 3.0])
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-len", type=int, default=96)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default="results/oracle/fuera_corpus/generaciones.csv")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def smiles_del_conjunto(nombre: str) -> list:
    """Todos los SMILES de un conjunto de MoleculeNet, de las tres particiones."""
    carpeta = ROOT / "data" / "deepchem_molnet" / DATASETS[nombre]["dir"] / "csv"
    if not carpeta.exists():
        raise SystemExit(f"no encuentro {carpeta}. Descarga los conjuntos antes.")
    trozos = []
    for parte in ("train", "valid", "test"):
        f = carpeta / f"{parte}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        col = next((c for c in d.columns if "smile" in c.lower()), None)
        if col is None:
            raise SystemExit(f"{f} no tiene columna de SMILES. Tiene: {list(d.columns)}")
        trozos.append(d[col].astype(str))
    if not trozos:
        raise SystemExit(f"no hay CSV bajo {carpeta}")
    return pd.concat(trozos).drop_duplicates().tolist()


def reparte_por_tamano(smiles: list, n: int, semilla: int) -> list:
    """n moléculas repartidas a lo largo del rango de átomos pesados del conjunto.

    Un muestreo uniforme daría sobre todo moléculas del centro de la distribución, que es
    justo donde la cobertura del corpus es mejor y donde la prueba exige menos. Se
    estratifica por tamaño para que los extremos estén representados.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    filas = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            filas.append((s, m.GetNumHeavyAtoms()))
    if not filas:
        return []
    d = pd.DataFrame(filas, columns=["smiles", "atomos"]).sort_values("atomos")
    if len(d) <= n:
        return d["smiles"].tolist()
    # Un estrato por molécula pedida, y una al azar dentro de cada estrato: tomar
    # siempre la primera de cada estrato daría el mismo sesgo dentro de cada tramo.
    rng = np.random.default_rng(semilla)
    cortes = np.array_split(np.arange(len(d)), n)
    idx = [int(rng.choice(c)) for c in cortes if len(c)]
    return d["smiles"].iloc[idx].tolist()


def main() -> None:
    args = parse_args()
    ruta = Path(args.ckpt)
    if not ruta.is_absolute():
        ruta = ROOT / ruta
    model, vocab, binners, ck = load_generator(ruta, args.device)
    char_vocab = ck.get("char_vocab")
    if char_vocab is None:
        raise SystemExit("el checkpoint no lleva char_vocab; no se puede featurizar igual "
                         "que en el entrenamiento")
    max_sm_len = int(ck.get("args", {}).get("max_sm_len", 100))
    binner = binners[args.property]
    prop_idx = PROPERTIES.index(args.property)
    null_row = [binners[n].null_bin for n in PROPERTIES]
    print(f"checkpoint {ruta.name} | brazo {ck.get('arm')} | paso {ck.get('step')}")

    # --- pertenencia al corpus -------------------------------------------------------
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    corpus = pd.read_csv(ROOT / args.corpus_dir / "corpus.csv")
    claves_corpus = set(corpus["inchikey"].astype(str))
    print(f"corpus: {len(claves_corpus):,} identificadores")

    partidas, origen = [], []
    for clave, bonito in CONJUNTOS.items():
        todos = smiles_del_conjunto(clave)
        fuera = []
        dentro = 0
        for s in todos:
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            if Chem.MolToInchiKey(m) in claves_corpus:
                dentro += 1
            else:
                fuera.append(s)
        elegidas = reparte_por_tamano(fuera, args.por_conjunto, args.seed)
        print(f"  {bonito:<14} {len(todos):>5} moléculas, {dentro:>4} también en el corpus, "
              f"se toman {len(elegidas)} de las {len(fuera)} restantes")
        partidas.extend(elegidas)
        origen.extend([bonito] * len(elegidas))

    if not partidas:
        raise SystemExit("ninguna molécula quedó fuera del corpus")

    # --- vocabulario de caracteres ---------------------------------------------------
    desconocidos = sorted({c for s in partidas for c in s if c not in char_vocab})
    afectadas = sum(any(c not in char_vocab for c in s) for s in partidas)
    if desconocidos:
        print(f"\n  [aviso] {len(desconocidos)} caracteres ausentes del vocabulario del "
              f"checkpoint: {desconocidos}")
        print(f"  afectan a {afectadas} de {len(partidas)} moléculas de partida, y en ellas "
              f"se codifican como relleno.")
    else:
        print("\n  todos los caracteres están en el vocabulario del checkpoint")

    largas = sum(len(s) > max_sm_len for s in partidas)
    if largas:
        print(f"  [aviso] {largas} moléculas superan los {max_sm_len} caracteres y la rama "
              f"de SMILES las ve truncadas")

    # --- generación ------------------------------------------------------------------
    cache = MoleculeGraphCache.build(partidas, char_vocab, max_sm_len=max_sm_len,
                                     workers=args.workers, schema="ogb")
    idx = np.arange(len(partidas))[cache.valid]
    if len(idx) < len(partidas):
        print(f"  {len(partidas) - len(idx)} moléculas que RDKit no pudo featurizar")

    props_partida = property_values([partidas[i] for i in idx])
    filas = []
    for w in args.guidance:
        for b in args.bins:
            cond = list(null_row)
            cond[prop_idx] = int(b)
            gen = generate_for_request(model, cache, idx, cond, vocab, args.device,
                                       args.batch_size, args.max_len, args.temperature,
                                       not args.greedy, w)
            canon, _ = canonical_and_key(gen)
            props_gen = property_values([c or "" for c in canon])
            centro = float(binner.bin_center(int(b)))
            for k, i in enumerate(idx):
                valido = canon[k] is not None
                filas.append({
                    "guidance": w, "conjunto": origen[i], "requested_bin": int(b),
                    "requested_centre": centro, "seed_row": int(i),
                    "seed_smiles": partidas[i], "seed_value": float(props_partida[k, prop_idx]),
                    "generated": canon[k] or "", "valid": valido,
                    "generated_value": float(props_gen[k, prop_idx]) if valido else float("nan"),
                    "obtained_delta": (float(props_gen[k, prop_idx] - props_partida[k, prop_idx])
                                       if valido else float("nan")),
                    "is_copy": bool(valido and canon[k] == partidas[i]),
                })
            print(f"  w={w} bin {b:>2}: {sum(c is not None for c in canon)}/{len(idx)} válidas",
                  flush=True)

    df = pd.DataFrame(filas)
    salida = ROOT / args.out
    salida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(salida, index=False)
    print(f"\nEscrito {salida}  ({len(df)} filas)")

    print(f"\n{'conjunto':<15}{'w':>5}{'n':>6}{'validez':>9}{'pendiente':>11}{'EAM':>8}")
    print("-" * 54)
    for (c, w), g in df.groupby(["conjunto", "guidance"]):
        ok = g[g["valid"] & np.isfinite(g["obtained_delta"])]
        if len(ok) < 3:
            print(f"{c:<15}{w:>5}{len(g):>6}{g['valid'].mean():>9.3f}   sin datos suficientes")
            continue
        pend = float(np.polyfit(ok["requested_centre"], ok["obtained_delta"], 1)[0])
        eam = float(np.mean(np.abs(ok["obtained_delta"] - ok["requested_centre"])))
        print(f"{c:<15}{w:>5}{len(g):>6}{g['valid'].mean():>9.3f}{pend:>11.3f}{eam:>8.3f}")
    print("\nLa pendiente es lo comparable con el Cuadro del oráculo. Una caída respecto de")
    print("aquella es el efecto que esta evaluación existe para medir.")


if __name__ == "__main__":
    main()
