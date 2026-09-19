"""Figuras del capítulo, en PDF vectorial.

    python scripts/figuras_tesis.py
    python scripts/figuras_tesis.py --moses-dir data/moses_solo --muestra 20000

Genera figuras/cobertura_distribucion.pdf: densidades del número de átomos pesados y del
peso molecular, corpus frente a los tres conjuntos de evaluación.

Sobre el corpus que se grafica. La figura sostiene la afirmación de que MOSES ocupa una
banda estrecha y falla por los dos extremos, de modo que graficar por error el corpus
ampliado con QM9 la contradiría sin que nada lo indicase. El script lee meta.json, avisa
si el corpus tiene más de una fuente y escribe en la propia figura cuántas moléculas se
muestrearon. Comprobar cuál se usó no debería requerir recordarlo.

La estimación de densidad es un núcleo gaussiano con la regla de Scott, implementado con
numpy: la biblioteca científica general no forma parte de las dependencias mínimas de
este trabajo y son diez líneas. Se muestrean --muestra moléculas por conjunto, las mismas
20 000 con que se calcularon los percentiles del texto, para que figura y tabla no puedan
discrepar.
"""
from __future__ import annotations

import argparse
import json
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

# Okabe-Ito, segura para las formas habituales de daltonismo. El corpus va en negro y con
# trazo grueso porque es la referencia contra la que se leen los otros tres, y cada curva
# lleva además un patrón de línea distinto para que la figura siga siendo legible
# impresa en escala de grises.
ESTILOS = {
    "MOSES":         dict(color="black",   lw=2.0, ls="-",  zorder=5),
    "ESOL":          dict(color="#0072B2", lw=1.3, ls="--", zorder=3),
    "FreeSolv":      dict(color="#E69F00", lw=1.3, ls="-.", zorder=3),
    "Lipophilicity": dict(color="#009E73", lw=1.3, ls=":",  zorder=3),
}

# El corpus de referencia: la banda sombreada es su rango intercuantílico 5-95 y las
# demás curvas se leen contra ella. Se dibuja para que el lector vea de un vistazo que
# FreeSolv queda entero por debajo y más de la mitad de Lipophilicity por encima, que es
# el argumento del apartado.
#
# La banda se calcula de la propia muestra graficada y no se fija a mano: un número
# escrito en el código puede acabar discrepando de los datos que la figura dibuja, y
# nadie lo notaría.
REFERENCIA = "MOSES"


def configurar_estilo() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # TrueType: el PDF queda con texto seleccionable y buscable, y no incrusta
        # subconjuntos Type 3 que algunos sistemas de impresión rechazan.
        "pdf.fonttype": 42,
        "figure.dpi": 150,
    })


def coma_decimal(x, _pos) -> str:
    """Separador decimal español, que es el que exige el resto del documento."""
    if x == int(x):
        return f"{int(x):d}".replace(",", ".")
    return f"{x:g}".replace(".", ",")


def kde(muestras: np.ndarray, malla: np.ndarray) -> np.ndarray:
    """Densidad por núcleo gaussiano con ancho de banda de Scott."""
    x = np.asarray(muestras, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return np.zeros_like(malla)
    bw = x.std(ddof=1) * n ** (-1.0 / 5.0)
    if bw <= 0:
        return np.zeros_like(malla)
    u = (malla[:, None] - x[None, :]) / bw
    return np.exp(-0.5 * u * u).sum(axis=1) / (n * bw * np.sqrt(2.0 * np.pi))


# --------------------------------------------------------------------------------------
# datos
# --------------------------------------------------------------------------------------

def descriptores(smiles, muestra: int, semilla: int = 0):
    """(átomos pesados, peso molecular) de una muestra aleatoria del conjunto."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors

    RDLogger.DisableLog("rdApp.*")
    s = list(smiles)
    if len(s) > muestra:
        idx = np.random.default_rng(semilla).choice(len(s), muestra, replace=False)
        s = [s[i] for i in idx]
    pesados, masa = [], []
    for x in s:
        m = Chem.MolFromSmiles(str(x))
        if m is not None:
            pesados.append(m.GetNumHeavyAtoms())
            masa.append(Descriptors.MolWt(m))
    return np.asarray(pesados, dtype=float), np.asarray(masa, dtype=float)


def cargar(args):
    """({nombre: (átomos pesados, peso molecular)}, {nombre: tamaño original}).

    La caché existe porque ajustar una figura son muchas ejecuciones y recalcular los
    descriptores de 80 000 moléculas en cada una convierte un retoque de treinta segundos
    en uno de un minuto. Lleva la huella de sus parámetros, de modo que cambiar el corpus
    o el tamaño de muestra la invalida en lugar de reutilizarla en silencio.
    """
    cache = ROOT / "figuras" / ".descriptores.npz"
    firma = f"{args.moses_dir}|{args.muestra}|{args.semilla}"
    if cache.exists() and not args.rehacer:
        z = np.load(cache, allow_pickle=False)
        if str(z["firma"]) == firma:
            print(f"  descriptores leídos de {cache.name}")
            return ({n: (z[f"{n}_pesados"], z[f"{n}_masa"]) for n in ESTILOS},
                    {n: int(z[f"{n}_total"]) for n in ESTILOS})

    moses_dir = ROOT / args.moses_dir
    meta = moses_dir / "meta.json"
    if meta.exists():
        fuentes = json.loads(meta.read_text(encoding="utf-8")).get("sources")
        if fuentes and len(fuentes) > 1:
            rutas = [f["path"] for f in fuentes]
            print(f"  AVISO: {args.moses_dir} tiene {len(fuentes)} fuentes: {rutas}")
            print("  La figura afirma que el corpus ocupa una banda estrecha; un corpus")
            print("  ampliado la contradice. Usa --moses-dir con el corpus sin ampliar.")

    conjuntos = {
        "MOSES": (moses_dir / "corpus.csv", "smiles"),
        "ESOL": (ROOT / "data/deepchem_molnet/delaney/csv", "smiles"),
        "FreeSolv": (ROOT / "data/deepchem_molnet/freesolv/csv", "smiles"),
        "Lipophilicity": (ROOT / "data/deepchem_molnet/lipo/csv", "smiles"),
    }

    datos, totales = {}, {}
    for nombre, (ruta, col) in conjuntos.items():
        if ruta.is_dir():
            # Los tres conjuntos de evaluación se grafican completos, recombinando sus
            # particiones: la figura describe el conjunto, no un subconjunto suyo.
            df = pd.concat([pd.read_csv(ruta / f"{p}.csv") for p in ("train", "valid", "test")],
                           ignore_index=True)
        elif ruta.exists():
            df = pd.read_csv(ruta, usecols=[col])
        else:
            raise SystemExit(f"no encuentro {ruta}")
        totales[nombre] = len(df)
        pesados, masa = descriptores(df[col].astype(str), args.muestra, args.semilla)
        datos[nombre] = (pesados, masa)
        print(f"  {nombre:<14} {len(pesados):>7,} moléculas  "
              f"átomos pesados p5/p50/p95 = "
              f"{np.percentile(pesados, 5):.0f}/{np.percentile(pesados, 50):.0f}/"
              f"{np.percentile(pesados, 95):.0f}  MW mediana {np.median(masa):.0f}")

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, firma=firma,
                        **{f"{n}_pesados": v[0] for n, v in datos.items()},
                        **{f"{n}_masa": v[1] for n, v in datos.items()},
                        **{f"{n}_total": totales[n] for n in datos})
    return datos, totales


# --------------------------------------------------------------------------------------
# figura 1
# --------------------------------------------------------------------------------------

def figura_cobertura(datos: dict, totales: dict, salida: Path,
                     png: bool = False) -> None:
    # 6.3 pulgadas es el \textwidth habitual de una tesis a una columna con márgenes de
    # 2,5 cm sobre A4. La altura se elige para que los paneles queden algo apaisados, que
    # es lo que conviene a una densidad.
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.3, 2.6))

    def panel(ax, indice, etiqueta_x, margen, decimales=0):
        """Dibuja un panel: banda del corpus, las cuatro densidades y la anotación."""
        valores = {n: v[indice] for n, v in datos.items()}
        todos = np.concatenate(list(valores.values()))
        malla = np.linspace(0, np.percentile(todos, 99.5) + margen, 400)

        lo, hi = np.percentile(valores[REFERENCIA], [5, 95])
        ax.axvspan(lo, hi, color="0.5", alpha=0.13, lw=0, zorder=0)
        for nombre, v in valores.items():
            ax.plot(malla, kde(v, malla), label=nombre, **ESTILOS[nombre])

        ax.set_xlabel(etiqueta_x)
        ax.set_ylabel("Densidad")
        ax.set_xlim(malla[0], malla[-1])
        # Un 22 % de aire por encima de la curva más alta, para que la anotación de la
        # banda no caiga sobre la densidad del corpus. Sin él, el texto y el pico de
        # MOSES ocupan el mismo sitio, que es justo donde la banda está.
        ax.set_ylim(0, max(kde(v, malla).max() for v in valores.values()) * 1.22)

        # La banda se anota dentro del panel: quien mire solo la figura no tiene por qué
        # saber qué significa el gris.
        fmt = f"{{:.{decimales}f}}"
        ax.annotate(f"{fmt.format(lo)}–{fmt.format(hi)}\n(p5–p95 {REFERENCIA})",
                    xy=((lo + hi) / 2, ax.get_ylim()[1] * 0.99),
                    ha="center", va="top", fontsize=6.5, color="0.35", linespacing=1.3)
        return lo, hi

    banda_a = panel(ax_a, 0, "Número de átomos pesados", margen=4)
    banda_b = panel(ax_b, 1, "Peso molecular (g/mol)", margen=40)
    print(f"  banda del corpus  átomos pesados {banda_a[0]:.0f}–{banda_a[1]:.0f}   "
          f"peso molecular {banda_b[0]:.0f}–{banda_b[1]:.0f}")

    for ax, etiqueta in ((ax_a, "(a)"), (ax_b, "(b)")):
        ax.text(-0.14, 1.04, etiqueta, transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="bottom", ha="left")
        ax.yaxis.set_major_formatter(FuncFormatter(coma_decimal))
        ax.xaxis.set_major_formatter(FuncFormatter(coma_decimal))
        ax.tick_params(length=2.5, pad=2)

    # Una sola leyenda para los dos paneles, bajo ellos, en una fila. Repetirla en cada
    # panel gastaría área de trazado en decir dos veces lo mismo.
    manejadores = [plt.Line2D([], [], **{k: v for k, v in ESTILOS[n].items() if k != "zorder"})
                   for n in ESTILOS]
    fig.legend(manejadores, list(ESTILOS), loc="lower center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, -0.06),
               handlelength=2.4, columnspacing=1.8)

    # El pie anterior decía "muestra de 20 000 por conjunto", que era falso: los tres
    # conjuntos de evaluación no llegan a ese tamaño y se grafican enteros. Se construye
    # de los datos, distinguiendo lo muestreado de lo completo.
    muestreados = [n for n in datos if totales.get(n, 0) > len(datos[n][0])]
    completos = [n for n in datos if n not in muestreados]
    partes = []
    if muestreados:
        partes.append("muestra de {} moléculas de {}".format(
            f"{len(datos[muestreados[0]][0]):,}".replace(",", " "),
            " y ".join(muestreados)))
    if completos:
        partes.append(("{} al completo" if len(completos) == 1 else "{}, al completo").format(
            ", ".join(completos)))
    # Solo la primera letra: str.capitalize pasa el resto a minúsculas y convertiría
    # MOSES y ESOL en moses y esol.
    pie = "; ".join(partes)
    fig.text(0.99, -0.055, pie[:1].upper() + pie[1:],
             ha="right", va="bottom", fontsize=6, color="0.45")

    fig.tight_layout()
    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida, bbox_inches="tight", format="pdf")
    print(f"\n  escrito {salida}")
    if png:
        # El PDF es lo que va al documento; el PNG existe para mirar la figura de un
        # vistazo cuando se genera en una máquina remota, donde ajustarla exige verla
        # varias veces. A 200 ppp basta para juzgar composición y legibilidad.
        alt = salida.with_suffix(".png")
        fig.savefig(alt, bbox_inches="tight", format="png", dpi=200)
        print(f"  escrito {alt}  (solo para previsualizar)")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--moses-dir", default="data/moses",
                   help="el corpus sin ampliar. Si tu data/moses lleva fuentes añadidas, "
                        "apunta a la copia previa")
    p.add_argument("--muestra", type=int, default=20_000,
                   help="moléculas por conjunto; las mismas con que se midieron los "
                        "percentiles del texto")
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--rehacer", action="store_true", help="ignorar los descriptores cacheados")
    p.add_argument("--salida", default="figuras/cobertura_distribucion.pdf")
    p.add_argument("--png", action="store_true",
                   help="ademas del PDF, un PNG a 200 ppp para previsualizar")
    args = p.parse_args()

    configurar_estilo()
    datos, totales = cargar(args)
    figura_cobertura(datos, totales, ROOT / args.salida, png=args.png)


if __name__ == "__main__":
    main()
