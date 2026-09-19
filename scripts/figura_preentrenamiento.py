"""Figura: efecto del preentrenamiento sobre el error de regresión, semilla a semilla.

    python scripts/figura_preentrenamiento.py
    python scripts/figura_preentrenamiento.py --fuente otro.csv --salida /tmp/f.pdf

Tres paneles, uno por conjunto. Cada semilla es un segmento que une su RMSE sin
preentrenar con el de la misma partición preentrenada, y sobre cada columna hay un trazo
en la media.

El gráfico es pareado a propósito. Con tres semillas, dos barras con su desviación no
distinguen "el preentrenamiento mejora" de "hay una partición fácil": ambas producen la
misma media y la misma dispersión. El segmento sí, porque muestra qué le ocurre a cada
partición por separado, y un descenso en las tres es un resultado que las barras esconden.

El estilo se importa de figuras_tesis.py en lugar de repetirse, para que las figuras del
capítulo no acaben con tipografías distintas.

Formato del CSV de entrada, una fila por (conjunto, semilla, pesos):

    conjunto,semilla,pesos,rmse
    ESOL,0,desde cero,0.680
    ESOL,0,preentrenado,0.571
    ...

No se interpola nada: si a alguna pareja le falta uno de los dos brazos, el script se
detiene y dice cuál.
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

FUENTE = ROOT / "results" / "rmse_por_semilla.csv"
SALIDA = ROOT / "figuras" / "efecto_preentrenamiento.pdf"

CONJUNTOS = ["ESOL", "FreeSolv", "Lipophilicity"]   # orden de los paneles
BRAZOS = ["desde cero", "preentrenado"]             # orden en el eje x

# Cifras publicadas en el texto. El script recalcula las suyas del CSV y avisa si
# discrepan: es la comprobación de que figura y tabla salen de los mismos números, que es
# justo el desajuste que nadie detecta leyendo.
CONTROL = {
    "ESOL":          {"desde cero": (0.741, 0.061), "preentrenado": (0.611, 0.040)},
    "FreeSolv":      {"desde cero": (1.420, 0.286), "preentrenado": (1.139, 0.136)},
    "Lipophilicity": {"desde cero": (0.845, 0.063), "preentrenado": (0.671, 0.026)},
}
TOLERANCIA = 0.0005    # media milésima: las cifras de control vienen a tres decimales

# ══════════════════════════════════════════════════════════════════════════════════════

COLOR_SEMILLA = "0.40"        # gris neutro: las semillas son el fondo, no el mensaje
COLOR_MEDIA = "#D55E00"       # Okabe-Ito, y el único acento que no usa la otra figura
ANCHO_MEDIA = 0.20            # media anchura del trazo de la media, en unidades de x

ANCHO_CM, ALTO_CM = 15.5, 5.5   # tamaño final del PDF ya recortado


def cargar(fuente: Path) -> pd.DataFrame:
    if not fuente.exists():
        raise SystemExit(
            f"no encuentro {fuente}\n"
            f"  Se espera un CSV con columnas: conjunto, semilla, pesos, rmse\n"
            f"  Edita FUENTE al principio del script, o pasa --fuente.")

    df = pd.read_csv(fuente)
    faltan = [c for c in ("conjunto", "semilla", "pesos", "rmse") if c not in df.columns]
    if faltan:
        raise SystemExit(f"{fuente.name} no tiene las columnas {faltan}. "
                         f"Tiene: {list(df.columns)}")

    df["conjunto"] = df["conjunto"].astype(str).str.strip()
    df["pesos"] = df["pesos"].astype(str).str.strip()
    df["rmse"] = pd.to_numeric(df["rmse"], errors="coerce")

    desconocidos = sorted(set(df["pesos"]) - set(BRAZOS))
    if desconocidos:
        raise SystemExit(f"valores de 'pesos' no reconocidos: {desconocidos}. "
                         f"Se esperaban exactamente {BRAZOS}.")

    # Una fila por (conjunto, semilla, pesos). Un duplicado promediaría en silencio dos
    # corridas distintas, que es peor que no dibujar nada.
    dup = df.duplicated(["conjunto", "semilla", "pesos"], keep=False)
    if dup.any():
        filas = df[dup].sort_values(["conjunto", "semilla", "pesos"])
        raise SystemExit("hay filas repetidas para la misma celda:\n"
                         + filas.to_string(index=False))

    problemas = []
    for conjunto in CONJUNTOS:
        sub = df[df["conjunto"] == conjunto]
        if sub.empty:
            problemas.append(f"  {conjunto}: no hay ninguna fila")
            continue
        for semilla in sorted(sub["semilla"].unique()):
            presentes = set(sub[sub["semilla"] == semilla]["pesos"])
            ausentes = [b for b in BRAZOS if b not in presentes]
            if ausentes:
                problemas.append(f"  {conjunto}, semilla {semilla}: "
                                 f"falta {', '.join(ausentes)}")
        for _, fila in sub[sub["rmse"].isna()].iterrows():
            problemas.append(f"  {conjunto}, semilla {fila['semilla']}, "
                             f"{fila['pesos']}: rmse no numérico")
    if problemas:
        raise SystemExit("el CSV está incompleto y la figura es pareada, "
                         "así que no se dibuja nada:\n" + "\n".join(problemas))

    sobran = sorted(set(df["conjunto"]) - set(CONJUNTOS))
    if sobran:
        print(f"  [aviso] se ignoran conjuntos no previstos: {sobran}")
    return df


def verificar(df: pd.DataFrame) -> None:
    """Medias y desviaciones por celda, contrastadas con las publicadas en el texto."""
    print(f"\n  {'conjunto':<15}{'pesos':<15}{'n':>3}{'media':>9}{'desv.':>8}   control")
    print("  " + "-" * 66)
    discrepan = []
    for conjunto in CONJUNTOS:
        for brazo in BRAZOS:
            v = df[(df["conjunto"] == conjunto) & (df["pesos"] == brazo)]["rmse"].to_numpy()
            media, desv = float(v.mean()), float(v.std(ddof=1))
            c_media, c_desv = CONTROL[conjunto][brazo]
            ok = abs(media - c_media) <= TOLERANCIA and abs(desv - c_desv) <= TOLERANCIA
            print(f"  {conjunto:<15}{brazo:<15}{len(v):>3}{media:>9.3f}{desv:>8.3f}   "
                  f"{c_media:.3f} +/- {c_desv:.3f}  {'ok' if ok else '<-- DISCREPA'}")
            if not ok:
                discrepan.append(f"{conjunto}/{brazo}")
    if discrepan:
        print(f"\n  [aviso] {len(discrepan)} celda(s) no coinciden con CONTROL: "
              f"{', '.join(discrepan)}")
        print("  O el CSV no es el de los resultados publicados, o el texto está sin")
        print("  actualizar. La figura se dibuja igualmente, con lo que dice el CSV.")


def ajustar_tamano(fig, ancho_cm: float, alto_cm: float, pasadas: int = 4) -> tuple:
    """Redimensiona el lienzo para que el PDF recortado mida lo pedido.

    bbox_inches="tight" recorta el sobrante, de modo que el figsize que se pasa a
    subplots no es el tamaño final: con estos paneles salían 15,1 x 3,6 cm en vez de
    15,5 x 5,5. Se mide el recuadro ajustado y se reescala el lienzo en esa proporción.
    Hacen falta varias pasadas porque las tipografías no escalan con él -- están en
    puntos -- y al cambiar el lienzo cambia la fracción que ocupan.
    """
    objetivo = (ancho_cm / 2.54, alto_cm / 2.54)
    ancho = alto = 0.0
    for _ in range(pasadas):
        fig.canvas.draw()
        caja = fig.get_tightbbox(fig.canvas.get_renderer())
        ancho, alto = caja.width, caja.height
        if ancho <= 0 or alto <= 0:
            break
        w, h = fig.get_size_inches()
        fig.set_size_inches(w * objetivo[0] / ancho, h * objetivo[1] / alto)
    return ancho * 2.54, alto * 2.54


def figura(df: pd.DataFrame, salida: Path, png: bool = True) -> None:
    fig, ejes = plt.subplots(1, 3, figsize=(ANCHO_CM / 2.54, ALTO_CM / 2.54))

    x = np.arange(len(BRAZOS), dtype=float)
    for k, (conjunto, ax) in enumerate(zip(CONJUNTOS, ejes)):
        sub = df[df["conjunto"] == conjunto]

        for semilla in sorted(sub["semilla"].unique()):
            fila = sub[sub["semilla"] == semilla].set_index("pesos")["rmse"]
            y = [float(fila[b]) for b in BRAZOS]
            ax.plot(x, y, color=COLOR_SEMILLA, alpha=0.6, lw=1.0,
                    marker="o", ms=3.4, mew=0, zorder=3)

        for xi, brazo in zip(x, BRAZOS):
            m = float(sub[sub["pesos"] == brazo]["rmse"].mean())
            ax.plot([xi - ANCHO_MEDIA, xi + ANCHO_MEDIA], [m, m],
                    color=COLOR_MEDIA, lw=2.4, solid_capstyle="butt", zorder=6)

        # Holgura vertical proporcional al rango dibujado, para que ningún marcador ni
        # trazo de media quede pegado al borde del panel.
        vals = sub["rmse"].to_numpy(dtype=float)
        lo, hi = float(vals.min()), float(vals.max())
        aire = max((hi - lo) * 0.14, 1e-3)
        ax.set_ylim(lo - aire, hi + aire)

        ax.set_xlim(-0.45, len(BRAZOS) - 1 + 0.45)
        ax.set_xticks(x)
        # Giradas: "desde cero" y "preentrenado" ocupan juntas más que el ancho de un
        # panel de 5 cm, y horizontales se solapan.
        ax.set_xticklabels(BRAZOS, rotation=20, ha="right", rotation_mode="anchor")
        ax.set_title(conjunto, fontsize=10, pad=5)
        ax.set_ylabel("RMSE", fontsize=9)
        ax.yaxis.set_major_formatter(FuncFormatter(coma_decimal))
        ax.tick_params(length=2.5, pad=2, labelsize=9)
        ax.grid(axis="y", color="0.88", lw=0.5, zorder=0)
        ax.set_axisbelow(True)

        # Por debajo de las etiquetas giradas, que descuelgan bastante del eje.
        ax.text(0.5, -0.62, f"({'abc'[k]})", transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="top", ha="center")

    fig.tight_layout(w_pad=1.8)
    medido = ajustar_tamano(fig, ANCHO_CM, ALTO_CM)
    print(f"\n  tamaño final del recorte: {medido[0]:.2f} x {medido[1]:.2f} cm "
          f"(objetivo {ANCHO_CM} x {ALTO_CM})")

    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida, bbox_inches="tight", pad_inches=0, format="pdf")
    print(f"\n  escrito {salida}")
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
    p.add_argument("--sin-png", action="store_true", help="solo el PDF")
    args = p.parse_args()

    configurar_estilo()
    df = cargar(Path(args.fuente))
    verificar(df)
    figura(df, Path(args.salida), png=not args.sin_png)


if __name__ == "__main__":
    main()
