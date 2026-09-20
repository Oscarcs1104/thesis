"""Figura: calibración del control de propiedad, pedido frente a obtenido.

    python scripts/figura_calibracion.py
    python scripts/figura_calibracion.py --fuente otro.csv --salida /tmp/c.pdf

Un panel, ejes con la misma escala y aspecto igual, de modo que la diagonal y = x cae a
45 grados. Esa diagonal es la obediencia exacta: cada punto por debajo de ella es una
petición que el generador se quedó corto en cumplir, y la distancia vertical a la
diagonal es, literalmente, el error de la petición.

El aspecto igual no es cosmético. Con ejes de escalas distintas, una pendiente de 0,4 se
puede dibujar pegada a la diagonal sin más que estirar el eje y, y la figura pasaría a
afirmar algo que los números no dicen.

Las dos series son dos pesos de guía sin clasificador. Comparar w = 1 con w = 3 en el
mismo panel es el diagnóstico: si subir el peso no despega los puntos de la horizontal,
la condición no se está usando.

El estilo se importa de figuras_tesis.py y el ajuste de tamaño de
figura_preentrenamiento.py, en lugar de repetirse, para que las figuras del capítulo no
acaben con tipografías ni márgenes distintos.

Formato del CSV de entrada, una fila por (intervalo, w):

    intervalo,w,delta_solicitado,delta_obtenido,desviacion
    1,1,-1.412,-0.783,0.612
    ...

El intervalo 0 es el nulo, la ausencia de condición, y no tiene posición en el eje x: se
descarta al leer. Nada se interpola: si a algún intervalo le falta su fila, el script se
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
from scripts.figura_preentrenamiento import ajustar_tamano  # noqa: E402

# ══════════════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN -- lo único que hay que tocar
# ══════════════════════════════════════════════════════════════════════════════════════

FUENTE = ROOT / "results" / "calibracion_logp.csv"
SALIDA = ROOT / "figuras" / "calibracion.pdf"

# Los intervalos que deben estar presentes para cada peso de guía. Si la evaluación
# numera sus intervalos desde 0 en lugar de desde 1, cámbialo aquí: el script no lo
# adivina, porque el intervalo 0 significa "sin condición" en una numeración y "primer
# cuantil" en la otra, y confundirlos borraría un intervalo real sin avisar.
INTERVALOS = list(range(1, 21))
NULO = 0                       # el intervalo sin condición, que no se representa

# clave en la columna w -> (etiqueta, color, marcador)
PESOS = {
    1: ("w = 1", "#0072B2", "o"),
    3: ("w = 3", "#D55E00", "s"),
}

ANCHO_CM, ALTO_CM = 12.0, 11.0   # tamaño final del PDF ya recortado

# ══════════════════════════════════════════════════════════════════════════════════════

COLUMNAS = ("intervalo", "w", "delta_solicitado", "delta_obtenido", "desviacion")


def cargar(fuente: Path) -> pd.DataFrame:
    if not fuente.exists():
        raise SystemExit(
            f"no encuentro {fuente}\n"
            f"  Se espera un CSV con columnas: {', '.join(COLUMNAS)}\n"
            f"  Edita FUENTE al principio del script, o pasa --fuente.")

    df = pd.read_csv(fuente)
    faltan = [c for c in COLUMNAS if c not in df.columns]
    if faltan:
        raise SystemExit(f"{fuente.name} no tiene las columnas {faltan}. "
                         f"Tiene: {list(df.columns)}")

    for c in COLUMNAS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    n_nulo = int((df["intervalo"] == NULO).sum())
    df = df[df["intervalo"] != NULO]
    if n_nulo:
        print(f"  descartadas {n_nulo} filas del intervalo {NULO} (sin condición)")

    df = df[df["w"].isin(PESOS)]
    if df.empty:
        raise SystemExit(f"no queda ninguna fila con w en {sorted(PESOS)}. "
                         f"Revisa la columna w del CSV.")

    dup = df.duplicated(["intervalo", "w"], keep=False)
    if dup.any():
        raise SystemExit("hay filas repetidas para el mismo (intervalo, w):\n"
                         + df[dup].sort_values(["w", "intervalo"]).to_string(index=False))

    problemas = []
    for w in PESOS:
        sub = df[df["w"] == w]
        if sub.empty:
            problemas.append(f"  w = {w}: no hay ninguna fila")
            continue
        ausentes = [i for i in INTERVALOS if i not in set(sub["intervalo"])]
        if ausentes:
            problemas.append(f"  w = {w}: faltan los intervalos {ausentes}")
        sobran = sorted(set(sub["intervalo"]) - set(INTERVALOS))
        if sobran:
            problemas.append(f"  w = {w}: intervalos fuera de {INTERVALOS[0]}..{INTERVALOS[-1]}: "
                             f"{sobran}. Si la evaluación numera desde 0, ajusta "
                             f"INTERVALOS al principio del script.")
        for _, fila in sub[sub[list(COLUMNAS)].isna().any(axis=1)].iterrows():
            problemas.append(f"  w = {w}, intervalo {fila['intervalo']}: valor no numérico")
    if problemas:
        # El caso que más fácil es confundir: si la evaluación numera desde 0, su primer
        # cuantil se ha descartado arriba tomándolo por el nulo, y lo único que se nota
        # es que "falta el último intervalo". Sin esta pista el aviso despista más que
        # ayuda, porque señala el extremo contrario al del problema.
        visto = set(df["intervalo"].dropna().astype(int))
        desplazado = set(range(INTERVALOS[0], INTERVALOS[-1]))
        if n_nulo and visto == desplazado:
            problemas.append(
                f"  Parece una numeración desde 0: se descartaron {n_nulo} filas con "
                f"intervalo {NULO} tomándolas por el nulo, y quedan justo "
                f"{INTERVALOS[0]}..{INTERVALOS[-1] - 1}. Si esos intervalos 0 eran el "
                f"primer cuantil y no la ausencia de condición, pon "
                f"INTERVALOS = list(range(0, {len(INTERVALOS)})) y NULO al valor que "
                f"use de verdad la evaluación.")
        raise SystemExit("el CSV está incompleto y no se dibuja nada:\n" + "\n".join(problemas))

    return df.sort_values(["w", "intervalo"]).reset_index(drop=True)


def estadisticos(x: np.ndarray, y: np.ndarray) -> tuple:
    """Pendiente y R2 de la regresión de obtenido sobre solicitado, y EAM a la diagonal.

    El R2 es el de esa regresión, no el de la diagonal: dice cuánto de la variación de lo
    obtenido explica la petición. Un modelo puede tener R2 alto y pendiente 0,3, es decir,
    obedecer de forma muy consistente pero muy atenuada, y ambas cifras hacen falta para
    describirlo. El EAM a la diagonal es lo que queda: el error en unidades de logP.
    """
    pendiente, corte = np.polyfit(x, y, 1)
    ajuste = pendiente * x + corte
    ss_res = float(np.sum((y - ajuste) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    eam = float(np.mean(np.abs(y - x)))
    return float(pendiente), float(corte), r2, eam


def verificar(df: pd.DataFrame) -> None:
    print(f"\n  {'peso':>6}{'n':>5}{'pendiente':>11}{'corte':>9}{'R2':>8}"
          f"{'EAM a la diagonal':>20}")
    print("  " + "-" * 59)
    for w, (etiqueta, _, _) in PESOS.items():
        sub = df[df["w"] == w]
        if sub.empty:
            continue
        x = sub["delta_solicitado"].to_numpy(dtype=float)
        y = sub["delta_obtenido"].to_numpy(dtype=float)
        pendiente, corte, r2, eam = estadisticos(x, y)
        print(f"  {etiqueta:>6}{len(sub):>5}{pendiente:>11.3f}{corte:>9.3f}"
              f"{r2:>8.3f}{eam:>20.3f}")
    print("\n  La pendiente es la obediencia: 1 sería control exacto, 0 indiferencia a la")
    print("  petición. El EAM está en unidades de logP y es la cifra interpretable.")


def figura(df: pd.DataFrame, salida: Path, png: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(ANCHO_CM / 2.54, ALTO_CM / 2.54))

    # Los límites se calculan antes de dibujar y con las barras de dispersión incluidas:
    # si se dejaran al autoescalado, cada eje elegiría el suyo y el aspecto igual dejaría
    # de significar 45 grados.
    lo = min(float(df["delta_solicitado"].min()),
             float((df["delta_obtenido"] - df["desviacion"]).min()))
    hi = max(float(df["delta_solicitado"].max()),
             float((df["delta_obtenido"] + df["desviacion"]).max()))
    aire = (hi - lo) * 0.06
    lo, hi = lo - aire, hi + aire

    # La diagonal, por debajo de todo: es la referencia, no un dato.
    ax.plot([lo, hi], [lo, hi], ls=":", color="0.55", lw=1.0, zorder=1)

    for w, (etiqueta, color, marcador) in PESOS.items():
        sub = df[df["w"] == w].sort_values("intervalo")
        if sub.empty:
            continue
        x = sub["delta_solicitado"].to_numpy(dtype=float)
        y = sub["delta_obtenido"].to_numpy(dtype=float)
        s = sub["desviacion"].to_numpy(dtype=float)

        ax.errorbar(x, y, yerr=s, fmt="none", ecolor=color, alpha=0.35,
                    elinewidth=0.8, capsize=1.8, capthick=0.8, zorder=2)
        ax.plot(x, y, color=color, alpha=0.45, lw=0.9, zorder=3)
        ax.plot(x, y, ls="none", marker=marcador, ms=4.0, mew=0,
                color=color, label=etiqueta, zorder=4)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Desplazamiento solicitado (logP)")
    ax.set_ylabel("Desplazamiento obtenido (logP)")
    ax.xaxis.set_major_formatter(FuncFormatter(coma_decimal))
    ax.yaxis.set_major_formatter(FuncFormatter(coma_decimal))
    ax.tick_params(length=2.5, pad=2, labelsize=9)
    ax.grid(color="0.90", lw=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Arriba a la izquierda: los datos siguen la diagonal, así que las dos esquinas
    # libres son esa y la de abajo a la derecha.
    leyenda = ax.legend(loc="upper left", fontsize=10, frameon=True,
                        framealpha=0.9, edgecolor="0.85", handletextpad=0.6)
    leyenda.get_frame().set_linewidth(0.5)

    fig.tight_layout()
    medido = ajustar_tamano(fig, ANCHO_CM, ALTO_CM)
    print(f"\n  tamaño final del recorte: {medido[0]:.2f} x {medido[1]:.2f} cm "
          f"(objetivo {ANCHO_CM} x {ALTO_CM})")

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
    p.add_argument("--sin-png", action="store_true", help="solo el PDF")
    args = p.parse_args()

    configurar_estilo()
    df = cargar(Path(args.fuente))
    verificar(df)
    figura(df, Path(args.salida), png=not args.sin_png)


if __name__ == "__main__":
    main()
