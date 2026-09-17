"""One command to read every result this pipeline produces, and say what is missing.

    python scripts/summarize_results.py

Reads whatever is on disk and reports it in the order the thesis argues it:

    1. predictive half   results/pretrained_ablation_*.csv
    2. generation, training   checkpoints/pairs/*_history.json
    3. generation, oracle     results/oracle/*/summary.json

Nothing here computes anything new; it only collects. The point is that "did it run,
and what did it say" should be one command rather than four globs and a pandas session,
and that a missing piece should be stated rather than inferred from an empty table.

Read the three sections with different weight. Section 1 is a benchmark number and
stands on its own. Section 2 is SELFIES reconstruction loss, which a model that copies
the seed scores well on -- mined pairs have Tanimoto >= 0.50 -- so it says the training
converged and nothing about conditioning. Section 3 is the one that answers the thesis
question, and until it exists the generation half has no result.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def predictive() -> bool:
    rule("1. Mitad predictiva -- ESOL / FreeSolv / Lipophilicity")
    files = sorted((ROOT / "results").glob("pretrained_ablation*.csv"))
    if not files:
        print("  (nada todavia: falta correr scripts/slurm/pretrained_ablation.sbatch)")
        return False

    import pandas as pd

    frames = []
    for f in files:
        df = pd.read_csv(f)
        # The filename carries which row this is when the CSV does not; _init vs _scratch
        # is the whole comparison, so guessing it wrong would invert the conclusion.
        df["source"] = f.stem.replace("pretrained_ablation", "").lstrip("_") or "default"
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    key = "pretrained" if "pretrained" in df.columns else "source"
    # pretrained is bool(args.init_checkpoint): whether the encoder started from the MOSES
    # checkpoint or from random init. True/False is what the CSV stores and a poor thing
    # to read in a table, so it is spelled out.
    label = {True: "preentrenado (MOSES)", False: "desde cero",
             "init": "preentrenado (MOSES)", "scratch": "desde cero"}
    # Rows measured on a frozen partition and on per-seed partitions answer different
    # questions and must not be pooled: the first has a spread covering initialisation
    # only, the second covers the partition too, and their means are not comparable.
    # config FIRST, and not optional. Averaging hybrid with mola and mola-fixed into one
    # "desde cero" row produced an n of 18 and three numbers that meant nothing: they
    # were the mean of three different architectures. Any column that distinguishes runs
    # has to split the table, never be averaged over.
    group_keys = ["dataset", "config"]
    for extra in ("split_protocol", "n_pool"):
        if extra in df.columns and df[extra].nunique() > 1:
            group_keys.append(extra)

    # Which encoder a run started from is a third axis, and it needs different handling
    # from the others: the from-scratch rows belong to no encoder and are the control for
    # all of them. Grouping on the column directly loses them twice over -- pandas reads
    # the empty tag as NaN and groupby drops NaN groups outright -- so instead the frame
    # is split per encoder with the shared control copied into each.
    if "init_tag" in df.columns:
        df["init_tag"] = df["init_tag"].fillna("").astype(str)
        encoders = sorted(t for t in df["init_tag"].unique() if t)
    else:
        encoders = []
    if len(encoders) > 1:
        scratch = df[df["init_tag"] == ""]
        blocks = [(df[df["init_tag"] == t]._append(scratch) if hasattr(df, "_append")
                   else pd.concat([df[df["init_tag"] == t], scratch]), t) for t in encoders]
    else:
        blocks = [(df, encoders[0] if encoders else None)]

    for frame, encoder in blocks:
      for keys, g in frame.groupby(group_keys):
        parts = [str(k) for k in (keys if isinstance(keys, tuple) else (keys,))]
        if encoder:
            parts.append(f"init={encoder}")
        print(f"\n  {' / '.join(parts)}")
        print(f"    {'fila':<22} {'RMSE':>16} {'MAE':>9} {'R2':>8} {'n':>3}")
        rows = {}
        for row_key, gg in g.groupby(key):
            rows[row_key] = gg.set_index("seed")["rmse"]
            print(f"    {label.get(row_key, str(row_key)):<22} "
                  f"{gg['rmse'].mean():>8.4f} +/- {gg['rmse'].std():<5.4f}"
                  f" {gg['mae'].mean():>8.4f} {gg['r2'].mean():>8.4f} {len(gg):>3}")
        # Whether the gap survives the seed spread is the whole question for this table,
        # so it is stated rather than left to the reader to eyeball two columns.
        if len(rows) == 2:
            (ka, a), (kb, b) = rows.items()
            shared = a.index.intersection(b.index)
            # The two rows run the same seeds, and under --resplit-per-seed that means the
            # same partitions. Comparing independent means throws that pairing away and
            # charges the partition variance to the error bar, where it cancels exactly:
            # under resplitting it is an order of magnitude larger than the effect, so the
            # unpaired test can call a real difference noise. Pair when the seeds line up.
            paired = len(shared) == len(a) == len(b) and len(shared) > 1
            if paired:
                d = (a.loc[shared] - b.loc[shared])
                gap, spread, how = d.mean(), d.std(), "pareada por semilla"
            else:
                gap = a.mean() - b.mean()
                spread = (a.std() + b.std()) / 2
                how = "entre medias (semillas no emparejadas)"
            better = label.get(ka if gap < 0 else kb, "?")
            if paired:
                # How often the difference points the same way, which the ratio cannot
                # say. The size of the benefit varies with the partition -- an easy test
                # set narrows it, a hard one widens it -- so the paired spread carries a
                # real interaction and not only noise, and a consistent sign is the
                # sturdier claim. n of n one way is p = 0.5**n under the null.
                wins = int((d < 0).sum()) if gap < 0 else int((d > 0).sum())
                p = 0.5 ** len(d)
                print(f"      -> {wins}/{len(d)} semillas favorecen a {better}"
                      + (f" (signo consistente, p={p:.3f} bajo la nula)" if wins == len(d) else ""))
            if abs(gap) < spread:
                print(f"      -> diferencia {abs(gap):.4f}, dentro de la dispersion "
                      f"{how} ({spread:.4f}): no es un resultado")
            else:
                # A ratio against a spread near zero is a number with no information in
                # it; say the spread vanished instead of printing seven digits of it.
                ratio = abs(gap) / spread if spread > 1e-6 else float("inf")
                size = ">100x" if ratio > 100 else f"{ratio:.1f}x"
                print(f"      -> {better} mejor por {abs(gap):.4f} RMSE, {size} la "
                      f"dispersion {how} ({spread:.4f})")
    # Encoder against encoder, paired by seed. Comparing them through their separate
    # blocks means comparing numbers from different runs, and two identical runs differ
    # by about 0.01 RMSE here: cuDNN picks algorithms by timing, PyG's scatter sums have
    # no fixed order, and early stopping turns that into a discrete choice of which epoch
    # wins. Pairing on the seed removes the partition draw; it does not remove that, so
    # the spread below still carries it and a difference under it is not a result.
    if len(encoders) > 1:
        rule("1b. Encoder contra encoder, pareado por semilla")
        pre = df[df["init_tag"] != ""]
        for dataset, g in pre.groupby("dataset"):
            series = {t: gg.set_index("seed")["rmse"] for t, gg in g.groupby("init_tag")}
            names = sorted(series)
            print(f"\n  {dataset}")
            for t in names:
                print(f"    {t:<40} {series[t].mean():.4f} +/- {series[t].std():.4f}")
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = series[names[i]], series[names[j]]
                    shared = a.index.intersection(b.index)
                    if len(shared) < 2:
                        print(f"    ({names[i]} vs {names[j]}: sin semillas en comun)")
                        continue
                    d = a.loc[shared] - b.loc[shared]
                    mejor = names[j] if d.mean() > 0 else names[i]
                    wins = int((d > 0).sum()) if d.mean() > 0 else int((d < 0).sum())
                    veredicto = ("dentro de la dispersion: no es un resultado"
                                 if abs(d.mean()) < d.std()
                                 else f"{abs(d.mean()) / max(d.std(), 1e-9):.1f}x la dispersion")
                    print(f"      -> {mejor} mejor por {abs(d.mean()):.4f} "
                          f"(+/- {d.std():.4f}), {wins}/{len(d)} semillas, {veredicto}")
        print("\n  Esta es la comparacion valida entre corpus de preentrenamiento. Leerla")
        print("  restando las medias de los bloques de arriba mezcla dos corridas distintas,")
        print("  y dos corridas identicas difieren aqui en torno a 0.01 RMSE.")

    print("\n  RMSE menor es mejor. La desviacion es sobre semillas. Con 3 semillas esto es")
    print("  una comprobacion de cordura, no una prueba estadistica: sirve para descartar")
    print("  diferencias que no existen, no para afirmar las que si.")
    return True


def generation_training() -> bool:
    rule("2. Generacion, entrenamiento -- NO responde la pregunta de la tesis")
    files = sorted((ROOT / "checkpoints" / "pairs").glob("*_history.json"))
    if not files:
        print("  (nada todavia: falta correr scripts/slurm/train_pairs.sbatch)")
        return False

    print(f"\n    {'corrida':<38} {'params':>11} {'pasos':>8} {'val loss':>9} "
          f"{'tok acc':>8} {'min':>5}")
    for f in files:
        blob = json.loads(f.read_text(encoding="utf-8"))
        # train_pairs.py writes {"arm", "params", "elapsed_s", "history"}; a bare list is
        # accepted too so an older or hand-made file does not crash the whole report.
        hist = blob.get("history", []) if isinstance(blob, dict) else blob
        meta = blob if isinstance(blob, dict) else {}
        evals = [h for h in hist if isinstance(h, dict) and "loss" in h]
        if not evals:
            print(f"    {f.stem.replace('_history', '')[:38]:<38} {'(sin evaluaciones)':>11}")
            continue
        best = min(evals, key=lambda h: h["loss"])
        print(f"    {f.stem.replace('_history', '')[:38]:<38} "
              f"{meta.get('params', 0):>11,} {evals[-1].get('step', 0):>8,} "
              f"{best['loss']:>9.4f} {best.get('token_acc', float('nan')):>8.4f} "
              f"{meta.get('elapsed_s', 0) / 60:>5.0f}")
    print("\n  Esto mide reconstruccion de SELFIES. Como los pares tienen Tanimoto >= 0.50,")
    print("  un modelo que copie la semilla puntua bien sin haber aprendido a condicionar.")
    print("  Sirve para ver que el entrenamiento convergio, no para comparar brazos.")
    print("\n  La columna de parametros si es comparable: los brazos de una sola modalidad")
    print("  deberian quedar cerca entre si. Si graph-only tiene ~450k y smiles-only ~3.9M,")
    print("  el checkpoint es anterior al equiparado de capacidad y la ablacion no es valida.")
    return True


def oracle() -> bool:
    rule("3. Generacion, oraculo -- ESTA es la pregunta de la tesis")
    files = sorted((ROOT / "results" / "oracle").glob("*/summary.json"))
    if not files:
        print("  (nada todavia: falta correr scripts/slurm/eval_oracle.sbatch por checkpoint)")
        print("\n  Sin esto la mitad generativa no tiene resultado. Las perdidas de la")
        print("  seccion 2 no distinguen un modelo que obedece de uno que copia.")
        return False

    print(f"\n    {'brazo':<30} {'w':>4} {'rho':>7} {'pend.':>7} {'valid':>6} "
          f"{'unico':>6} {'novel':>6} {'copia':>6} {'tanim':>6} {'nulo':>7} {'sd':>6}")
    for f in files:
        s = json.loads(f.read_text(encoding="utf-8"))
        name = f.parent.name
        for w, g in sorted(s.get("guidance", {}).items(), key=lambda kv: float(kv[0])):
            nc = g.get("null_control", {})
            print(f"    {name[:30]:<30} {float(w):>4.1f} "
                  f"{g['spearman_request_vs_obtained']:>+7.3f} "
                  f"{g['slope_obtained_per_requested']:>+7.3f} "
                  f"{g['validity']:>6.3f} {g['uniqueness']:>6.3f} {g['novelty']:>6.3f} "
                  f"{g['copy_rate']:>6.3f} {g['mean_tanimoto_to_seed']:>6.3f} "
                  f"{nc.get('mean_delta', float('nan')):>+7.3f} "
                  f"{nc.get('std_delta', float('nan')):>6.3f}")
            name = ""      # solo en la primera fila de cada brazo
    print("\n  rho   correlacion entre el bin pedido y el delta obtenido, intra-semilla.")
    print("        Es el numero. Un modelo que ignora la condicion da ~0 aunque su")
    print("        perdida de la seccion 2 sea la mejor de las cuatro.")
    print("  pend. delta obtenido por unidad de delta pedido; 1.0 seria obediencia exacta.")
    print("  copia fraccion que devuelve la semilla sin cambios. rho alto con copia alta")
    print("        no existe, pero copia alta con rho ~0 es el modo de fallo esperado.")
    print("\n  Si rho no sube al subir w, la guia libre de clasificador no esta haciendo")
    print("  nada, que es el diagnostico gratis que esa perilla compra.")
    print("\n  nulo  el mismo brazo, mismas semillas, condicion puesta al bin nulo. Es la")
    print("        referencia honesta: si su delta medio y su dispersion se parecen a los")
    print("        de los bins pedidos, la condicion no esta haciendo el trabajo y rho")
    print("        viene de otra parte. Debe rondar cero con dispersion ancha.")
    return True


def main() -> None:
    done = [predictive(), generation_training(), oracle()]
    rule("Estado")
    labels = ["mitad predictiva", "generacion (entrenamiento)", "generacion (oraculo)"]
    for ok, label in zip(done, labels):
        print(f"  [{'x' if ok else ' '}] {label}")
    if not done[2]:
        print("\n  Siguiente: una corrida de oraculo por checkpoint.")
        ckpts = sorted((ROOT / "checkpoints" / "pairs").glob("*.pt"))
        for c in ckpts:
            print(f"    CKPT=checkpoints/pairs/{c.name} sbatch scripts/slurm/eval_oracle.sbatch")
        if not ckpts:
            print("    (no hay checkpoints de pares todavia)")
    sys.exit(0)


if __name__ == "__main__":
    main()
