"""Figura: una misma molécula de partida sometida a cinco peticiones distintas.

    python scripts/figura_rejilla.py --corrida pairs_graph_smiles_pretrained_s2025_logp
    python scripts/figura_rejilla.py --corrida ... --semilla 711169 --guia 3

La figura recorre el rango admitido de izquierda a derecha: se pide reducir logP, se pide
no cambiarlo y se pide aumentarlo, siempre desde la misma molécula. Es la forma más
directa de mostrar que la condición hace algo, porque lo único que varía entre paneles es
lo que se pidió.

Qué semilla se dibuja. Por omisión, la de comportamiento mediano entre las cien, medida
como el error absoluto medio de sus generaciones en los cinco intervalos representados.
No la mejor: una rejilla elegida por ser la que mejor obedece enseña el mejor caso
disponible sin decirlo, y el lector no tiene forma de saberlo. Con --semilla se fuerza
otra, y el script imprime en qué percentil queda la elegida para que la decisión sea
explícita.

La molécula de partida se dibuja como primer panel, separada del resto. Sin ella no se
puede ver qué cambió, que es justamente lo que la figura quiere mostrar.
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.figuras_tesis import configurar_estilo  # noqa: E402
from scripts.figura_preentrenamiento import ajustar_tamano  # noqa: E402

# ══════════════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════════════

# Los cinco intervalos representados, simétricos alrededor del centro: los dos extremos,
# dos intermedios y el que pide no cambiar nada.
INTERVALOS = [0, 5, 10, 14, 19]

SALIDA = ROOT / "figuras" / "rejilla_desplazamientos.pdf"
ANCHO_CM, ALTO_CM = 15.5, 5.0

PX = 380            # lado en píxeles de cada estructura antes de componerla
COLOR_PARTIDA = "#0072B2"

# ══════════════════════════════════════════════════════════════════════════════════════


def dibujar(smiles: str, px: int = PX):
    """Estructura como matriz RGB. Cairo si está, y si no el respaldo de PIL."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise SystemExit(f"RDKit no puede leer este SMILES: {smiles}")
    Chem.rdDepictor.Compute2DCoords(mol)

    try:
        d = rdMolDraw2D.MolDraw2DCairo(px, px)
        d.drawOptions().clearBackground = False
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
        d.FinishDrawing()
        return plt.imread(io.BytesIO(d.GetDrawingText()), format="png")
    except Exception:
        return np.asarray(Draw.MolToImage(mol, size=(px, px))) / 255.0


def coma(x: float, dec: int = 2, signo: bool = True) -> str:
    """Coma decimal y signo menos tipografico (U+2212), no el guion del teclado.

    Las etiquetas de los ejes de las demas figuras ya usan ese caracter, que es el que
    matplotlib pone por defecto en los numeros negativos; escribir aqui un guion dejaria
    dos signos distintos en el mismo capitulo.
    """
    s = f"{x:+.{dec}f}" if signo else f"{x:.{dec}f}"
    return s.replace(".", ",").replace("-", "−")


def cargar(ruta: Path, guia: float) -> pd.DataFrame:
    if not ruta.exists():
        raise SystemExit(f"no encuentro {ruta}\n"
                         f"  Pasa --corrida con el nombre del directorio bajo results/oracle/")
    # keep_default_na=False: pandas convierte la cadena "null" de `request` en NaN y las
    # filas del control desaparecen sin aviso.
    df = pd.read_csv(ruta, keep_default_na=False, na_values=[""])
    if guia not in set(df["guidance"]):
        raise SystemExit(f"la corrida no tiene guía {guia}. "
                         f"Tiene: {sorted(df['guidance'].unique())}")
    df = df[(df["guidance"] == guia) & (df["request"] == "bin")]
    falta = [b for b in INTERVALOS if b not in set(df["requested_bin"])]
    if falta:
        raise SystemExit(f"la corrida no tiene los intervalos {falta}. "
                         f"Ajusta INTERVALOS al principio del script.")
    return df[df["requested_bin"].isin(INTERVALOS)].copy()


def elegir_semilla(df: pd.DataFrame, forzada: int | None) -> int:
    """La semilla mediana por error absoluto medio sobre los cinco intervalos."""
    df = df.copy()
    df["error"] = (df["obtained_delta"] - df["requested_centre"]).abs()
    # Solo se consideran semillas con generación válida en los cinco intervalos: una
    # rejilla a la que le falte un panel no sirve.
    completas = df[df["valid"].astype(bool)].groupby("seed_row")["error"]
    tabla = completas.agg(["mean", "count"])
    tabla = tabla[tabla["count"] == len(INTERVALOS)].sort_values("mean")
    if tabla.empty:
        raise SystemExit("ninguna semilla tiene generación válida en los cinco intervalos")

    if forzada is not None:
        if forzada not in tabla.index:
            raise SystemExit(f"la semilla {forzada} no tiene generación válida en los "
                             f"cinco intervalos. Hay {len(tabla)} que sí.")
        pos = int(tabla.index.get_loc(forzada))
        print(f"  semilla {forzada} forzada: percentil {100 * pos / len(tabla):.0f} "
              f"de {len(tabla)} (0 = la que mejor obedece)")
        return forzada

    pos = len(tabla) // 2
    elegida = int(tabla.index[pos])
    print(f"  {len(tabla)} semillas completas; se toma la mediana: fila {elegida}, "
          f"error medio {tabla['mean'].iloc[pos]:.3f} "
          f"(mejor {tabla['mean'].iloc[0]:.3f}, peor {tabla['mean'].iloc[-1]:.3f})")
    return elegida


def figura(df: pd.DataFrame, semilla: int, salida: Path, png: bool = True) -> None:
    sub = df[df["seed_row"] == semilla].set_index("requested_bin")
    partida = sub.iloc[0]["seed_smiles"]
    logp_partida = float(sub.iloc[0]["seed_value"])

    fig, ejes = plt.subplots(1, len(INTERVALOS) + 1,
                             figsize=(ANCHO_CM / 2.54, ALTO_CM / 2.54))

    ejes[0].imshow(dibujar(partida))
    ejes[0].set_title("Partida", fontsize=8, pad=3, color=COLOR_PARTIDA)
    ejes[0].set_xlabel(f"logP {coma(logp_partida, 2, False)}", fontsize=7, labelpad=3)
    for lado in ejes[0].spines.values():
        lado.set_color(COLOR_PARTIDA)
        lado.set_linewidth(0.8)

    print(f"\n  partida  {partida}   logP {logp_partida:.3f}")
    for ax, b in zip(ejes[1:], INTERVALOS):
        fila = sub.loc[b]
        ax.imshow(dibujar(str(fila["generated"])))
        ax.set_title(f"pedido {coma(float(fila['requested_centre']))}",
                     fontsize=8, pad=3)
        ax.set_xlabel(f"obtenido {coma(float(fila['obtained_delta']))}",
                      fontsize=7, labelpad=3)
        for lado in ax.spines.values():
            lado.set_color("0.80")
            lado.set_linewidth(0.5)
        print(f"  bin {b:>2}  pedido {fila['requested_centre']:+.3f}  "
              f"obtenido {fila['obtained_delta']:+.3f}  "
              f"T {fila['tanimoto_to_seed']:.3f}  {fila['generated']}")

    for ax in ejes:
        ax.set_xticks([])
        ax.set_yticks([])
        # El estilo del capitulo apaga los bordes superior y derecho, que es lo correcto
        # en un grafico de ejes pero deja estos paneles con medio marco.
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(True)
            ax.spines[lado].set_color(ax.spines["left"].get_edgecolor())
            ax.spines[lado].set_linewidth(ax.spines["left"].get_linewidth())

    fig.tight_layout(w_pad=0.6)
    medido = ajustar_tamano(fig, ANCHO_CM, ALTO_CM)
    print(f"\n  tamaño final del recorte: {medido[0]:.2f} x {medido[1]:.2f} cm")

    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida, bbox_inches="tight", pad_inches=0, format="pdf")
    print(f"  escrito {salida}")
    if png:
        alt = salida.with_suffix(".png")
        fig.savefig(alt, bbox_inches="tight", pad_inches=0, format="png", dpi=300)
        print(f"  escrito {alt}")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corrida", required=True)
    p.add_argument("--guia", type=float, default=1.0)
    p.add_argument("--semilla", type=int, default=None,
                   help="fila de la molécula de partida; por omisión, la mediana")
    p.add_argument("--salida", default=str(SALIDA))
    p.add_argument("--sin-png", action="store_true")
    args = p.parse_args()

    configurar_estilo()
    df = cargar(ROOT / "results" / "oracle" / args.corrida / "generations.csv", args.guia)
    figura(df, elegir_semilla(df, args.semilla), Path(args.salida), png=not args.sin_png)


if __name__ == "__main__":
    main()
