"""Figura: generación condicionada desde moléculas ajenas al corpus de preentrenamiento.

    python scripts/figura_generalizacion.py
    python scripts/figura_generalizacion.py --bin 19 --guia 3

Una fila por conjunto de evaluación. En cada una, la molécula de partida, la generada y
el desplazamiento pedido frente al obtenido. Los tres conjuntos ocupan tramos distintos
del rango de tamaños, de modo que las filas se leen de arriba abajo como un recorrido de
molécula pequeña a molécula grande, que es el eje sobre el que la cobertura del corpus
falla.

Qué molécula se dibuja de cada conjunto: la de error mediano en ese intervalo, por la
misma razón que en la rejilla. La mejor de cada conjunto haría parecer que el control
sobrevive intacto fuera del corpus, que es justo lo que la figura debe permitir juzgar.

Lee el CSV que produce crossmodal_model/generation/eval_fuera_corpus.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.figuras_tesis import configurar_estilo  # noqa: E402
from scripts.figura_preentrenamiento import ajustar_tamano  # noqa: E402
from scripts.figura_flujo import cambiados, coma, dibujar  # noqa: E402

FUENTE = ROOT / "results" / "oracle" / "fuera_corpus" / "generaciones.csv"
SALIDA = ROOT / "figuras" / "generalizacion_fuera_corpus.pdf"

ORDEN = ["FreeSolv", "ESOL", "Lipophilicity"]   # de moléculas pequeñas a grandes
ANCHO_CM, ALTO_CM = 15.5, 13.0
COLOR_TEXTO = "#0072B2"

# Pixeles por enlace, iguales en los tres paneles. Con el escalado automatico de RDKit
# una molecula de cuatro atomos llena su recuadro igual que una de treinta, y la figura
# dejaria de mostrar lo unico que la justifica: que los conjuntos ocupan tramos distintos
# del rango de tamanos.
ENLACE_PX = 26.0


def cargar(fuente: Path, guia: float, b: int) -> pd.DataFrame:
    if not fuente.exists():
        raise SystemExit(
            f"no encuentro {fuente}\n"
            f"  Genera antes con:\n"
            f"    python -m crossmodal_model.generation.eval_fuera_corpus --ckpt <ckpt>")
    df = pd.read_csv(fuente, keep_default_na=False, na_values=[""])
    hay = sorted(df["guidance"].unique()), sorted(df["requested_bin"].unique())
    df = df[(df["guidance"] == guia) & (df["requested_bin"] == b) & df["valid"].astype(bool)]
    if df.empty:
        raise SystemExit(f"no hay filas válidas con guía {guia} e intervalo {b}. "
                         f"El CSV tiene guías {hay[0]} e intervalos {hay[1]}.")
    faltan = [c for c in ORDEN if c not in set(df["conjunto"])]
    if faltan:
        raise SystemExit(f"no hay generaciones válidas de {faltan} en ese intervalo")
    df = df.copy()
    df["error"] = (df["obtained_delta"] - df["requested_centre"]).abs()
    return df


def elige(df: pd.DataFrame, conjunto: str):
    """La generación de error mediano del conjunto, con desempate determinista."""
    sub = df[df["conjunto"] == conjunto].sort_values(["error", "seed_row"])
    return sub.iloc[len(sub) // 2]


def figura(df: pd.DataFrame, salida: Path, png: bool = True) -> None:
    from rdkit import Chem

    fig = plt.figure(figsize=(ANCHO_CM / 2.54, ALTO_CM / 2.54))
    rejilla = fig.add_gridspec(len(ORDEN), 3, width_ratios=[3.0, 3.0, 1.9],
                               wspace=0.05, hspace=0.18)

    for f, conjunto in enumerate(ORDEN):
        fila = elige(df, conjunto)
        partida, generada = str(fila["seed_smiles"]), str(fila["generated"])
        a, b = Chem.MolFromSmiles(partida), Chem.MolFromSmiles(generada)
        fuera, dentro = cambiados(b, a), cambiados(a, b)
        if fuera is None or dentro is None:
            fuera = dentro = []

        ax_a = fig.add_subplot(rejilla[f, 0])
        ax_b = fig.add_subplot(rejilla[f, 1])
        ax_t = fig.add_subplot(rejilla[f, 2])

        ax_a.imshow(dibujar(partida, fuera, longitud_enlace=ENLACE_PX))
        ax_b.imshow(dibujar(generada, dentro, longitud_enlace=ENLACE_PX))
        # El nombre del conjunto va como rótulo del eje y de la primera columna: es la
        # etiqueta de la fila, no un título, y arriba competiría con las cabeceras.
        ax_a.set_ylabel(conjunto, fontsize=9, labelpad=6, color=COLOR_TEXTO)
        if f == 0:
            ax_a.set_title("Molécula de partida", fontsize=9, pad=4)
            ax_b.set_title("Molécula generada", fontsize=9, pad=4)

        ax_t.text(0.5, 0.66, f"pedido\n$\\Delta$ logP {coma(float(fila['requested_centre']))}",
                  ha="center", va="center", fontsize=8.5, linespacing=1.5,
                  transform=ax_t.transAxes)
        ax_t.text(0.5, 0.30, f"obtenido\n$\\Delta$ logP {coma(float(fila['obtained_delta']))}",
                  ha="center", va="center", fontsize=8.5, linespacing=1.5,
                  color=COLOR_TEXTO, transform=ax_t.transAxes)

        for ax in (ax_a, ax_b):
            ax.set_xticks([]); ax.set_yticks([])
            for lado in ("top", "right", "bottom", "left"):
                ax.spines[lado].set_visible(True)
                ax.spines[lado].set_color("0.80")
                ax.spines[lado].set_linewidth(0.5)
        ax_t.set_axis_off()

        print(f"  {conjunto:<14} {a.GetNumHeavyAtoms():>3} átomos  "
              f"pedido {fila['requested_centre']:+.3f}  "
              f"obtenido {fila['obtained_delta']:+.3f}")

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
    p.add_argument("--fuente", default=str(FUENTE))
    p.add_argument("--salida", default=str(SALIDA))
    p.add_argument("--guia", type=float, default=1.0)
    p.add_argument("--bin", type=int, default=19,
                   help="intervalo solicitado que se ilustra")
    p.add_argument("--sin-png", action="store_true")
    args = p.parse_args()

    configurar_estilo()
    figura(cargar(Path(args.fuente), args.guia, args.bin), Path(args.salida),
           png=not args.sin_png)


if __name__ == "__main__":
    main()
