"""Figura: distribución del desplazamiento entre los dos miembros de cada par análogo.

    python scripts/figura_deltas.py
    python scripts/figura_deltas.py --muestra 2000000 --png

Cuatro paneles en rejilla 2x2, uno por propiedad, cada uno con el histograma de la
diferencia entre la molécula de llegada y la de partida. Las dos líneas discontinuas son
los percentiles 1 y 99, que delimitan el rango de desplazamiento que la evaluación puede
solicitar sin extrapolar.

El eje x se fuerza simétrico alrededor de cero. No es una decisión estética: los pares se
emiten en ambas direcciones, de modo que la simetría de cada histograma es una propiedad
que el minado debe cumplir, y un eje descentrado la ocultaría.

El estilo se importa de figuras_tesis.py en lugar de repetirse, para que las dos figuras
del capítulo no acaben con tipografías distintas.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.figuras_tesis import coma_decimal, configurar_estilo  # noqa: E402

# ══════════════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN -- lo único que hay que tocar
# ══════════════════════════════════════════════════════════════════════════════════════

# De dónde salen los desplazamientos. Se admiten tres formas:
#   .npy      matriz [P, 4] en el orden de PROPIEDADES (lo que produce mine_pairs.py)
#   .csv      una columna por propiedad, nombradas según las claves de PROPIEDADES
#   .parquet  igual que el CSV
FUENTE = ROOT / "data" / "moses" / "pair_deltas.npy"

# clave en los datos -> título del panel, en el orden de la rejilla (arriba-izquierda,
# arriba-derecha, abajo-izquierda, abajo-derecha)
PROPIEDADES = {
    "logp": "logP",
    "tpsa": "TPSA",
    "qed":  "QED",
    "mw":   "MW (g/mol)",
}

SALIDA = ROOT / "figuras" / "distribucion_deltas.pdf"

# ══════════════════════════════════════════════════════════════════════════════════════

N_BINS = 120
COLOR_BARRA = "0.62"
COLOR_LINEA = "#0072B2"       # Okabe-Ito, la misma paleta que la otra figura


def cargar_deltas(fuente: Path, muestra: int | None, semilla: int = 0) -> pd.DataFrame:
    """DataFrame con una columna por propiedad. Sustituir si los datos vienen de otro sitio.

    El muestreo existe porque el corpus tiene del orden de trece millones de pares y
    ajustar una figura son muchas ejecuciones; con dos millones el histograma ya no se
    distingue del completo.
    """
    if not fuente.exists():
        raise SystemExit(f"no encuentro {fuente}\n"
                         f"  Edita FUENTE al principio del script, o pasa --fuente.")

    if fuente.suffix == ".npy":
        arr = np.load(fuente, mmap_mode="r")
        if arr.ndim != 2 or arr.shape[1] != len(PROPIEDADES):
            raise SystemExit(f"{fuente.name} tiene forma {arr.shape}; se esperaban "
                             f"{len(PROPIEDADES)} columnas, una por propiedad")
        filas = np.arange(arr.shape[0])
        if muestra and arr.shape[0] > muestra:
            filas = np.sort(np.random.default_rng(semilla).choice(
                arr.shape[0], muestra, replace=False))
        df = pd.DataFrame(np.asarray(arr[filas]), columns=list(PROPIEDADES))
    else:
        df = (pd.read_parquet(fuente) if fuente.suffix == ".parquet"
              else pd.read_csv(fuente))
        faltan = [c for c in PROPIEDADES if c not in df.columns]
        if faltan:
            raise SystemExit(f"{fuente.name} no tiene las columnas {faltan}. "
                             f"Tiene: {list(df.columns)[:10]}\n"
                             f"  Ajusta las claves de PROPIEDADES al principio del script.")
        df = df[list(PROPIEDADES)]
        if muestra and len(df) > muestra:
            df = df.sample(muestra, random_state=semilla)

    print(f"  {len(df):,} pares leídos de {fuente.name}".replace(",", " "))
    return df


def formato(valor: float) -> str:
    """Decimales según la magnitud: logP y QED necesitan tres, MW con tres sería ruido."""
    a = abs(valor)
    dec = 3 if a < 1 else (2 if a < 10 else 1)
    return f"{valor:+.{dec}f}".replace(".", ",")


def figura_deltas(df: pd.DataFrame, salida: Path, png: bool = False) -> None:
    fig, ejes = plt.subplots(2, 2, figsize=(6.3, 4.4))

    for k, ((clave, titulo), ax) in enumerate(zip(PROPIEDADES.items(), ejes.flat)):
        x = df[clave].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        p1, p99 = np.percentile(x, [1, 99])

        # Simétrico y algo más ancho que los percentiles dibujados, para que las líneas
        # no queden pegadas al borde y se vea que las colas continúan.
        limite = max(abs(p1), abs(p99)) * 1.35
        ax.hist(x, bins=N_BINS, range=(-limite, limite), density=True,
                color=COLOR_BARRA, edgecolor="none", zorder=2)

        for v in (p1, p99):
            ax.axvline(v, color=COLOR_LINEA, ls="--", lw=1.0, zorder=3)

        ax.set_xlim(-limite, limite)
        # Un 18 % de aire arriba: las etiquetas de los percentiles van ahí y sobre las
        # barras serían ilegibles.
        techo = ax.get_ylim()[1]
        ax.set_ylim(0, techo * 1.18)

        ax.annotate(f"p1\n{formato(p1)}", xy=(p1, techo * 1.16), xytext=(-3, 0),
                    textcoords="offset points", ha="right", va="top",
                    fontsize=6.5, color=COLOR_LINEA, linespacing=1.25)
        ax.annotate(f"p99\n{formato(p99)}", xy=(p99, techo * 1.16), xytext=(3, 0),
                    textcoords="offset points", ha="left", va="top",
                    fontsize=6.5, color=COLOR_LINEA, linespacing=1.25)

        ax.set_title(titulo, fontsize=9, pad=4)
        ax.xaxis.set_major_formatter(FuncFormatter(coma_decimal))
        ax.yaxis.set_major_formatter(FuncFormatter(coma_decimal))
        ax.tick_params(length=2.5, pad=2)

        # Rótulos solo en el borde de la rejilla: repetirlos en los cuatro paneles gasta
        # espacio en decir cuatro veces lo mismo.
        if k % 2 == 0:
            ax.set_ylabel("Densidad")
        if k >= 2:
            ax.set_xlabel("Desplazamiento $\\Delta$")

        print(f"  {titulo:<12} media {np.mean(x):+.4f}  desv. típica {np.std(x, ddof=1):.4f}  "
              f"p1 {p1:+.4f}  p99 {p99:+.4f}")

    fig.tight_layout()
    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida, bbox_inches="tight", format="pdf")
    print(f"\n  escrito {salida}")
    if png:
        alt = salida.with_suffix(".png")
        fig.savefig(alt, bbox_inches="tight", format="png", dpi=200)
        print(f"  escrito {alt}  (solo para previsualizar)")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fuente", default=str(FUENTE))
    p.add_argument("--salida", default=str(SALIDA))
    p.add_argument("--muestra", type=int, default=2_000_000,
                   help="pares a leer; 0 para todos")
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--png", action="store_true",
                   help="además del PDF, un PNG a 200 ppp para previsualizar")
    args = p.parse_args()

    configurar_estilo()
    df = cargar_deltas(Path(args.fuente), args.muestra or None, args.semilla)
    figura_deltas(df, Path(args.salida), png=args.png)


if __name__ == "__main__":
    main()
