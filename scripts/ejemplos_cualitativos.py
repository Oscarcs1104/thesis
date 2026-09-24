"""Selecciona los cuatro ejemplos cualitativos y emite el Cuadro 13 en LaTeX.

    python scripts/ejemplos_cualitativos.py --corrida pairs_graph_smiles_pretrained_s2025_logp
    python scripts/ejemplos_cualitativos.py --corrida ... --guia 3

Los cuatro casos son los que el capítulo describe: dos aciertos en direcciones opuestas,
una copia y una petición en el extremo del rango. La selección se hace aquí, con reglas
escritas, y no a ojo sobre el CSV: elegir los ejemplos mirando cuál queda bonito es
seleccionar sobre el resultado, y el lector no tiene forma de saber que ocurrió.

Imprime además el SMILES de partida y el generado de cada ejemplo, que es lo que las
figuras del flujo y de la rejilla necesitan dibujar. Así los cuatro ejemplos del cuadro y
los de las figuras son forzosamente los mismos.

Sobre la lectura del CSV: pandas convierte la cadena "null" de la columna `request` en
NaN, de modo que las 100 filas del control nulo desaparecen sin aviso. De ahí el
keep_default_na=False. Aquí solo se usan las filas condicionadas, pero el recuento que se
imprime al principio sirve de comprobación de que el fichero se leyó entero.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Un desplazamiento pequeño se cumple por casualidad con demasiada facilidad, así que los
# dos ejemplos de acierto se buscan entre las peticiones que exigen un cambio real.
UMBRAL_PETICION = 0.5


def coma(x: float, dec: int = 3, signo: bool = False) -> str:
    """Número en notación española, listo para modo matemático de LaTeX."""
    s = f"{x:+.{dec}f}" if signo else f"{x:.{dec}f}"
    return s.replace(".", "{,}")


def cargar(ruta: Path, guia: float) -> pd.DataFrame:
    if not ruta.exists():
        raise SystemExit(f"no encuentro {ruta}\n"
                         f"  Pasa --corrida con el nombre del directorio bajo results/oracle/")

    df = pd.read_csv(ruta, keep_default_na=False, na_values=[""])
    print(f"  {len(df)} filas, de las cuales {int((df['request'] == 'null').sum())} "
          f"del control nulo")

    pesos = sorted(df["guidance"].unique())
    if guia not in pesos:
        raise SystemExit(f"la corrida no tiene guía {guia}. Tiene: {pesos}")

    df = df[(df["guidance"] == guia) & (df["request"] == "bin") & df["valid"].astype(bool)]
    if df.empty:
        raise SystemExit(f"no quedan filas válidas con guía {guia}")

    df = df.copy()
    df["error"] = (df["obtained_delta"] - df["requested_centre"]).abs()
    df["acierto"] = (df["obtained_delta"] > df["bin_lo"]) & (df["obtained_delta"] <= df["bin_hi"])
    df["copia"] = df["is_copy"].astype(bool)
    return df


def elegir(df: pd.DataFrame) -> list:
    """Los cuatro ejemplos, por reglas fijas y con desempate determinista.

    El desempate es siempre por (error, seed_row): sin él, dos ejecuciones sobre el mismo
    CSV podrían devolver filas distintas según el orden interno de pandas, y el cuadro de
    la memoria dejaría de ser reproducible.
    """
    casos = []

    grande = df[df["requested_centre"].abs() >= UMBRAL_PETICION]

    def mejor(sub, descripcion):
        if sub.empty:
            return None
        fila = sub.sort_values(["error", "seed_row"]).iloc[0]
        return (descripcion, fila)

    # 1 y 2: el acierto más ajustado en cada dirección, descartando las copias, que se
    # tratan aparte y que en una petición grande serían un fallo, no un ejemplo.
    abajo = mejor(grande[(grande["requested_centre"] < 0) & ~grande["copia"]],
                  "acierto, petición de reducir logP")
    arriba = mejor(grande[(grande["requested_centre"] > 0) & ~grande["copia"]],
                   "acierto, petición de aumentar logP")

    # 3: una copia. Se busca en los intervalos centrales, donde devolver la molécula de
    # partida es la respuesta correcta y no un fallo del condicionamiento.
    copias = df[df["copia"]].copy()
    copias["centralidad"] = copias["requested_centre"].abs()
    copia = None
    if not copias.empty:
        fila = copias.sort_values(["centralidad", "seed_row"]).iloc[0]
        copia = ("copia, petición próxima a cero", fila)

    # 4: el extremo del rango. Se toma el intervalo más alejado de cero que exista y,
    # dentro de él, la generación mediana en error, no la mejor: el ejemplo debe
    # representar el comportamiento típico en el extremo, no su mejor caso.
    extremos = df[df["requested_centre"].abs() == df["requested_centre"].abs().max()]
    extremo = None
    if not extremos.empty:
        ordenado = extremos.sort_values(["error", "seed_row"])
        fila = ordenado.iloc[len(ordenado) // 2]
        extremo = ("extremo del rango, generación típica", fila)

    for c in (abajo, arriba, copia, extremo):
        if c is not None:
            casos.append(c)
    if len(casos) < 4:
        print(f"  [aviso] solo se encontraron {len(casos)} de los 4 ejemplos previstos")
    return casos


def emitir(casos: list, guia: float) -> None:
    print("\n" + "=" * 78)
    print("SMILES de cada ejemplo (para las Figuras del flujo y de la rejilla)")
    print("=" * 78)
    for i, (desc, f) in enumerate(casos, 1):
        print(f"\nEjemplo {i}: {desc}")
        print(f"  intervalo {int(f['requested_bin'])}  fila semilla {int(f['seed_row'])}")
        print(f"  partida   {f['seed_smiles']}   logP {f['seed_value']:.3f}")
        print(f"  generada  {f['generated']}   logP {f['generated_value']:.3f}")
        print(f"  pedido {f['requested_centre']:+.3f}  obtenido {f['obtained_delta']:+.3f}"
              f"  Tanimoto {f['tanimoto_to_seed']:.3f}"
              f"  {'acierto' if f['acierto'] else 'fuera del intervalo'}")

    print("\n" + "=" * 78)
    print("Cuadro 13 en LaTeX")
    print("=" * 78 + "\n")
    print("\\begin{table}[htbp]")
    print("  \\centering")
    print("  \\small")
    print("  \\begin{tabular}{@{}lrrrr@{}}")
    print("    \\toprule")
    print("    Ejemplo & $\\Delta$ solicitado & $\\Delta$ obtenido & Tanimoto & Acierto \\\\")
    print("    \\midrule")
    for i, (_, f) in enumerate(casos, 1):
        print(f"    Ejemplo {i} & ${coma(f['requested_centre'], 3, True)}$ "
              f"& ${coma(f['obtained_delta'], 3, True)}$ "
              f"& {f['tanimoto_to_seed']:.3f}".replace(".", ",") + " "
              f"& {'sí' if f['acierto'] else 'no'} \\\\")
    print("    \\bottomrule")
    print("  \\end{tabular}")
    print(f"  \\caption{{Métricas de los ejemplos cualitativos seleccionados, "
          f"con peso de guía $w = {int(guia)}$.}}")
    print("  \\label{tab:ejemplos-cualitativos}")
    print("\\end{table}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corrida", required=True,
                   help="nombre del directorio bajo results/oracle/")
    p.add_argument("--guia", type=float, default=1.0)
    args = p.parse_args()

    ruta = ROOT / "results" / "oracle" / args.corrida / "generations.csv"
    df = cargar(ruta, args.guia)
    emitir(elegir(df), args.guia)


if __name__ == "__main__":
    main()
