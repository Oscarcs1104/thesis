"""Fetch a SMILES-only corpus and report what of the evaluation sets it reaches.

    python data_pipeline/download_corpus.py chembl
    python data_pipeline/download_corpus.py chembl --profile-only --sample 50000
    python data_pipeline/download_corpus.py --url <...> --smiles-col canonical_smiles --name mine

Profile first. The QM9 attempt is the argument for it: QM9 covers FreeSolv's size regime
exactly, and adding it changed nothing there, because after deduplication it was 6.3% of
the union and the encoder saw 640k of its molecules against 9.6M of MOSES's. Coverage
that does not weigh enough to shift the training distribution does not transfer, so the
question to settle before downloading gigabytes is not only "does this reach the gap" but
"how large a share of the corpus would it be".

What the measured gaps are (heavy atoms, p5/p50/p95):

    MOSES         17 / 21 / 25    the corpus: ZINC Clean Leads filtered to MW 250-350
    ZINC-250k     18 / 22 / 25    the same band; scaling ZINC adds volume, not coverage
    FreeSolv       4 /  8 / 18    entirely below
    ESOL           4 / 12 / 25    half below
    Lipophilicity 15 / 27 / 38    more than half above

So the corpus misses both ends, and nothing currently reaches the upper one. ChEMBL is
the obvious candidate there: it is medicinal chemistry without MOSES's molecular-weight
filter, and Lipophilicity is drawn from it.

    --profile-only  streams the first --sample molecules, prints the distribution beside
                    the evaluation sets, and writes nothing. Minutes instead of an hour.

URLs change between releases. The presets below are starting points, not guarantees;
if one 404s, check the project's download page and pass --url.
"""
from __future__ import annotations

import argparse
import gzip
import io
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# name -> (url, smiles column, separator). Verify before trusting: ChEMBL puts its
# release number in the filename and PubChem rebuilds its extras periodically.
PRESETS = {
    "chembl": (
        "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_35_chemreps.txt.gz",
        "canonical_smiles", "\t",
    ),
    "pubchem": (
        "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz",
        None, "\t",           # no header: CID <tab> SMILES
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("preset", nargs="?", choices=sorted(PRESETS),
                   help="a known source; omit it and pass --url for anything else")
    p.add_argument("--url", help="overrides the preset's URL, or supplies one")
    p.add_argument("--local", help="a file already on disk instead of a download")
    p.add_argument("--smiles-col", help="column name, or an integer index for headerless files")
    p.add_argument("--sep", default=None, help="field separator (default: the preset's)")
    p.add_argument("--name", help="output goes to data/<name>.csv (default: the preset)")
    p.add_argument("--sample", type=int, default=50_000,
                   help="molecules to read for --profile-only")
    p.add_argument("--limit", type=int, default=None,
                   help="keep only the first N molecules. Use it to size the corpus "
                        "deliberately: a source that ends up a small share of the union "
                        "cannot shift what the encoder learns, whatever it covers")
    p.add_argument("--profile-only", action="store_true",
                   help="measure the distribution and write nothing")
    return p.parse_args()


def stream_smiles(url: str | None, local: str | None, col, sep: str, limit: int | None):
    """Yield SMILES without holding the whole file in memory. These sources are large."""
    if local:
        raw = open(local, "rb")
    else:
        print(f"Streaming {url}")
        raw = urllib.request.urlopen(url)
    with raw:
        stream = gzip.GzipFile(fileobj=raw) if (local or url).endswith(".gz") else raw
        text = io.TextIOWrapper(stream, encoding="utf-8", errors="replace")
        header = None
        idx = None
        for n, line in enumerate(text):
            parts = line.rstrip("\n").split(sep)
            if n == 0:
                looks_like_header = isinstance(col, str) and col in parts
                if looks_like_header:
                    header, idx = parts, parts.index(col)
                    continue
                # Headerless, or a column given by position. PubChem's CID-SMILES is
                # "<cid>\t<smiles>" with no header at all.
                idx = int(col) if (col is not None and str(col).isdigit()) else len(parts) - 1
            if idx is None or idx >= len(parts):
                continue
            s = parts[idx].strip()
            if s:
                yield s
            if limit and n + 1 >= limit:
                return


def main() -> None:
    args = parse_args()
    url, col, sep = (PRESETS.get(args.preset) or (None, None, "\t"))
    url = args.url or url
    col = args.smiles_col or col
    sep = args.sep or sep
    name = args.name or args.preset or "corpus"
    if not (url or args.local):
        raise SystemExit("give a preset, --url or --local")

    limit = args.sample if args.profile_only else args.limit
    smiles = list(stream_smiles(url, args.local, col, sep, limit))
    print(f"read {len(smiles):,} SMILES")
    if not smiles:
        raise SystemExit("nothing was read -- check --smiles-col and --sep against the file")
    print(f"  first three: {smiles[:3]}")

    from data_pipeline.download_qm9 import profile_everything

    profile_everything(smiles)

    # The share this source would end up as, which is the part the QM9 attempt got wrong.
    corpus_csv = ROOT / "data" / "moses" / "corpus.csv"
    if corpus_csv.exists() and not args.profile_only:
        current = sum(1 for _ in open(corpus_csv, encoding="utf-8")) - 1
        share = len(smiles) / (current + len(smiles))
        print(f"\n  peso en el corpus unido: {len(smiles):,} de {current + len(smiles):,} "
              f"= {100 * share:.1f}%")
        if share < 0.15:
            print("  AVISO: por debajo del 15%, esta fuente difícilmente movera lo que el")
            print("  encoder aprende. QM9 entro al 6,3% y no cambio nada donde debia.")

    if args.profile_only:
        print(f"\nSolo perfil sobre {len(smiles):,} moleculas; no se ha escrito nada.")
        print("Quita --profile-only para descargarlo entero.")
        return

    out = ROOT / "data" / f"{name}.csv"
    df = pd.DataFrame({"smiles": smiles}).drop_duplicates(subset="smiles")
    df.to_csv(out, index=False)
    print(f"\nWrote {out}: {len(df):,} unique SMILES")
    print(f"Next:  python data_pipeline/moses.py --extra-csv data/{name}.csv --force")


if __name__ == "__main__":
    main()
