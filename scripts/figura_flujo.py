"""Figura: el flujo de una generación condicionada, de la molécula de partida a la obtenida.

    python scripts/figura_flujo.py --corrida pairs_graph_smiles_pretrained_s2025_logp
    python scripts/figura_flujo.py --corrida ... --ejemplo 4

Cuatro bloques en fila: la molécula de partida con su logP, el desplazamiento solicitado,
la molécula generada con su logP y el desplazamiento obtenido.

Qué ejemplo se dibuja. Los mismos cuatro que selecciona ejemplos_cualitativos.py, y por
las mismas reglas, importadas de allí en lugar de repetidas: si el cuadro y la figura
eligieran por su cuenta, podrían acabar ilustrando moléculas distintas y nadie lo notaría.
Por omisión se toma el segundo, que es un acierto con un cambio estructural localizado.

Cómo se marca lo que cambió. Por subestructura común máxima: se calcula lo que las dos
moléculas comparten y se resalta el resto. La alternativa, la diferencia de huellas
circulares, marca el vecindario entero de cada átomo alterado y sobre estos ejemplos pinta
entre el 50 y el 100 por ciento de la molécula, frente al 8-27 por ciento del MCS. Lo que
un químico llama "qué cambió" es lo segundo.

El MCS puede agotar su tiempo en moléculas grandes. Si eso ocurre no se inventa un
resaltado aproximado: se dibuja sin resaltar y se avisa por consola, porque una figura que
señala el fragmento equivocado es peor que una que no señala ninguno.
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

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.figuras_tesis import configurar_estilo  # noqa: E402
from scripts.figura_preentrenamiento import ajustar_tamano  # noqa: E402
from scripts.ejemplos_cualitativos import cargar, elegir  # noqa: E402

SALIDA = ROOT / "figuras" / "flujo_generacion.pdf"
ANCHO_CM, ALTO_CM = 15.5, 5.5

PX = 520
COLOR_TEXTO = "#0072B2"
RESALTE = (1.00, 0.80, 0.63)      # tinte del acento #D55E00, legible bajo los enlaces
SEG_MCS = 20                      # tiempo máximo del MCS


def coma(x: float, dec: int = 2, signo: bool = True) -> str:
    s = f"{x:+.{dec}f}" if signo else f"{x:.{dec}f}"
    return s.replace(".", ",").replace("-", "−")


def cambiados(a, b):
    """Átomos de b que no están en la subestructura común con a. None si el MCS falla."""
    from rdkit import Chem
    from rdkit.Chem import rdFMCS

    r = rdFMCS.FindMCS([a, b], timeout=SEG_MCS,
                       ringMatchesRingOnly=True, completeRingsOnly=True)
    if r.canceled or not r.smartsString:
        return None
    patron = Chem.MolFromSmarts(r.smartsString)
    if patron is None:
        return None
    comunes = b.GetSubstructMatch(patron)
    if not comunes:
        return None
    return sorted(set(range(b.GetNumAtoms())) - set(comunes))


def dibujar(smiles: str, resaltar=None, px: int = PX, longitud_enlace: float = 0.0):
    """Estructura como matriz RGB.

    longitud_enlace fija los pixeles por enlace en lugar de dejar que RDKit escale cada
    molecula hasta llenar su recuadro. Sin ella, una molecula de cuatro atomos y una de
    treinta salen del mismo tamano, lo que da igual cuando se comparan dos analogos y
    borra el argumento cuando lo que se compara son conjuntos de tamanos distintos.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise SystemExit(f"RDKit no puede leer este SMILES: {smiles}")
    Chem.rdDepictor.Compute2DCoords(mol)
    resaltar = list(resaltar or [])

    try:
        d = rdMolDraw2D.MolDraw2DCairo(px, px)
        d.drawOptions().clearBackground = False
        if longitud_enlace > 0:
            d.drawOptions().fixedBondLength = longitud_enlace
        rdMolDraw2D.PrepareAndDrawMolecule(
            d, mol, highlightAtoms=resaltar,
            highlightAtomColors={i: RESALTE for i in resaltar})
        d.FinishDrawing()
        return plt.imread(io.BytesIO(d.GetDrawingText()), format="png")
    except Exception:
        return np.asarray(Draw.MolToImage(mol, size=(px, px),
                                          highlightAtoms=resaltar)) / 255.0


def figura(fila, descripcion: str, salida: Path, png: bool = True) -> None:
    from rdkit import Chem

    partida, generada = str(fila["seed_smiles"]), str(fila["generated"])
    a, b = Chem.MolFromSmiles(partida), Chem.MolFromSmiles(generada)

    # El resaltado se calcula en las dos direcciones: en la de partida, lo que
    # desapareció; en la generada, lo que apareció. Marcar solo una de las dos deja al
    # lector adivinando de dónde salió el fragmento nuevo.
    fuera = cambiados(b, a)
    dentro = cambiados(a, b)
    if fuera is None or dentro is None:
        print("  [aviso] el MCS no terminó; se dibuja sin resaltar")
        fuera = dentro = []
    print(f"  resaltados: {len(fuera)}/{a.GetNumAtoms()} en la de partida, "
          f"{len(dentro)}/{b.GetNumAtoms()} en la generada")

    fig = plt.figure(figsize=(ANCHO_CM / 2.54, ALTO_CM / 2.54))
    rejilla = fig.add_gridspec(1, 4, width_ratios=[3.0, 1.5, 3.0, 1.5], wspace=0.05)
    ax_a, ax_flecha, ax_b, ax_res = [fig.add_subplot(rejilla[0, i]) for i in range(4)]

    ax_a.imshow(dibujar(partida, fuera))
    ax_a.set_title("Molécula de partida", fontsize=8, pad=4)
    ax_a.set_xlabel(f"logP {coma(float(fila['seed_value']), 2, False)}",
                    fontsize=8, labelpad=4)

    ax_b.imshow(dibujar(generada, dentro))
    ax_b.set_title("Molécula generada", fontsize=8, pad=4)
    ax_b.set_xlabel(f"logP {coma(float(fila['generated_value']), 2, False)}",
                    fontsize=8, labelpad=4)

    # Bloque 2: la petición, sobre la flecha que va de una molécula a la otra.
    ax_flecha.annotate("", xy=(0.92, 0.5), xytext=(0.08, 0.5),
                       xycoords="axes fraction",
                       arrowprops=dict(arrowstyle="-|>", color=COLOR_TEXTO, lw=1.2,
                                       shrinkA=0, shrinkB=0))
    ax_flecha.text(0.5, 0.60, "se solicita", ha="center", va="bottom", fontsize=8,
                   transform=ax_flecha.transAxes)
    ax_flecha.text(0.5, 0.34, f"$\\Delta$ logP {coma(float(fila['requested_centre']))}",
                   ha="center", va="top", fontsize=9, color=COLOR_TEXTO,
                   transform=ax_flecha.transAxes)

    # Bloque 4: lo obtenido, medido con el mismo oráculo sobre la molécula generada.
    ax_res.text(0.5, 0.60, "se obtiene", ha="center", va="bottom", fontsize=8,
                transform=ax_res.transAxes)
    ax_res.text(0.5, 0.34, f"$\\Delta$ logP {coma(float(fila['obtained_delta']))}",
                ha="center", va="top", fontsize=9, color=COLOR_TEXTO,
                transform=ax_res.transAxes)
    ax_res.text(0.5, 0.14, f"Tanimoto {coma(float(fila['tanimoto_to_seed']), 3, False)}",
                ha="center", va="top", fontsize=7.5, color="0.35",
                transform=ax_res.transAxes)

    for ax in (ax_a, ax_b):
        ax.set_xticks([]); ax.set_yticks([])
        for lado in ax.spines.values():
            lado.set_visible(True); lado.set_color("0.80"); lado.set_linewidth(0.5)
    for ax in (ax_flecha, ax_res):
        ax.set_axis_off()

    print(f"  ejemplo: {descripcion}")
    print(f"  partida   {partida}")
    print(f"  generada  {generada}")

    # Sin tight_layout: dos de los cuatro bloques no tienen ejes y no sabe medirlos, de
    # modo que avisa y recoloca peor de lo que ya lo hace el gridspec.
    medido = ajustar_tamano(fig, ANCHO_CM, ALTO_CM)
    print(f"  tamaño final del recorte: {medido[0]:.2f} x {medido[1]:.2f} cm")

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
    p.add_argument("--ejemplo", type=int, default=2, choices=(1, 2, 3, 4),
                   help="cuál de los cuatro ejemplos cualitativos se dibuja")
    p.add_argument("--salida", default=str(SALIDA))
    p.add_argument("--sin-png", action="store_true")
    args = p.parse_args()

    configurar_estilo()
    ruta = ROOT / "results" / "oracle" / args.corrida / "generations.csv"
    casos = elegir(cargar(ruta, args.guia))
    if args.ejemplo > len(casos):
        raise SystemExit(f"solo se encontraron {len(casos)} ejemplos; "
                         f"se pidió el {args.ejemplo}")
    descripcion, fila = casos[args.ejemplo - 1]
    figura(fila, descripcion, Path(args.salida), png=not args.sin_png)


if __name__ == "__main__":
    main()
