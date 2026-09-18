"""Add IntDiv1 and IntDiv2 to oracle runs that were evaluated before the metric existed.

    python scripts/add_diversity.py

The generated molecules are in results/oracle/*/generations.csv, so the metric can be
computed on runs already finished rather than regenerating them. No GPU, no model.

    IntDiv_p(G) = 1 - mean_i ( mean_j T(m_i, m_j)^p ) ^ (1/p)

Two readings are written, and they answer different questions for a conditional
generator. Globally, over every request at one guidance weight, diversity is inflated by
the requests themselves: asking for bin 0 and bin 19 produces different molecules by
design, so a model that merely responds to the condition already scores well. Within a
bin -- the same delta asked of a hundred different seeds -- it measures what is wanted,
how varied the answers to one request are.

It complements the two diversity numbers already reported. Uniqueness counts distinct
outputs and treats two molecules differing by one methyl as entirely different; novelty
asks whether an output appears in the training corpus and says nothing about how the
outputs relate to each other. Neither notices a generator whose outputs are all distinct
and all nearly the same.

Existing fields in summary.json are left alone; only the diversity ones are added or
replaced, so rerunning is safe.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def main() -> None:
    # Imported from the evaluator rather than reimplemented, so the numbers added here
    # and the ones future runs compute cannot drift apart.
    from crossmodal_model.generation.eval_oracle import internal_diversity

    runs = sorted((ROOT / "results" / "oracle").glob("*/summary.json"))
    if not runs:
        raise SystemExit("no oracle runs under results/oracle/")

    print(f"{'corrida':<46}{'w':>5}{'IntDiv1':>9}{'IntDiv2':>9}"
          f"{'IntDiv1':>10}{'IntDiv2':>9}")
    print(f"{'':46}{'':5}{'global':>9}{'global':>9}{'por bin':>10}{'por bin':>9}")
    for summary_path in runs:
        gen_path = summary_path.parent / "generations.csv"
        if not gen_path.exists():
            print(f"  {summary_path.parent.name}: no generations.csv, saltada")
            continue
        df = pd.read_csv(gen_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        name = summary_path.parent.name

        for w_key, block in summary.get("guidance", {}).items():
            gw = df[(df["guidance"] == float(w_key)) & df["valid"]]
            conditioned = gw[gw["request"] == "bin"]
            if conditioned.empty:
                continue

            block["intdiv1"] = internal_diversity(conditioned["generated"].tolist(), p=1)
            block["intdiv2"] = internal_diversity(conditioned["generated"].tolist(), p=2)
            for q in (1, 2):
                vals = []
                for _, gb in conditioned.groupby("requested_bin"):
                    v = internal_diversity(gb["generated"].tolist(), p=q)
                    if v == v:                       # not NaN
                        vals.append(v)
                block[f"intdiv{q}_within_bin"] = sum(vals) / len(vals) if vals else float("nan")

            print(f"  {name[:44]:<44}{float(w_key):>5.1f}"
                  f"{block['intdiv1']:>9.3f}{block['intdiv2']:>9.3f}"
                  f"{block['intdiv1_within_bin']:>10.3f}{block['intdiv2_within_bin']:>9.3f}")
            name = ""

        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n  Escrito en cada summary.json. Lee la columna 'por bin': la global esta")
    print("  inflada por las propias peticiones, que producen moleculas distintas aposta.")


if __name__ == "__main__":
    main()
