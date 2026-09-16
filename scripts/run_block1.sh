#!/usr/bin/env bash
# Rebuild every data artifact on a machine without SLURM. The sbatch equivalent is
# scripts/slurm/block1_data.sbatch; this is the same four stages in the same order.
#
#   nohup bash scripts/run_block1.sh > results/logs/block1.log 2>&1 &
#   tail -f results/logs/block1.log
#
#   LIMIT=5000 bash scripts/run_block1.sh      # smoke: minutes instead of hours
#
# Everything here is deterministic given seed 2025 and the same downloads, so running
# it on a second machine reproduces the first machine's corpus rather than approximating
# it. That is why the data is regenerated instead of copied: about 2 GB of transfer
# against a few hours of CPU that needs no supervision.
#
# The order is not interchangeable. moses.py excludes from the corpus every molecule
# that appears in ESOL / FreeSolv / Lipophilicity, so the MoleculeNet splits have to
# exist before it runs -- otherwise the deduplication silently has nothing to match
# against and the corpus keeps the evaluation molecules.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/logs

if [[ -f "${THESIS_VENV:-$HOME/venvs/thesis}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${THESIS_VENV:-$HOME/venvs/thesis}/bin/activate"
fi
# Each stage forks one process per core already; BLAS threads on top of that
# oversubscribe the machine and make it slower.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1

# Leave a couple of cores for the machine to stay usable -- this runs for hours.
W="${WORKERS:-$(( $(nproc) > 4 ? $(nproc) - 2 : 1 ))}"
LIMIT_ARG=${LIMIT:+--limit $LIMIT}
F=${FORCE:+--force}

python -c "import rdkit, selfies, numpy, pandas"   # fail in seconds on a bad env
echo "== $W workers =="

step() {
  echo
  echo "=== $1 ($(date +%H:%M:%S)) ==="
  shift
  "$@"
}

step "1/4 MoleculeNet splits (needed by the corpus deduplication)" \
  python data_pipeline/prepare_all.py --split "${SPLIT:-random}" --seed "${SEED:-2025}" --skip-zinc
step "2/4 MOSES corpus" \
  python data_pipeline/moses.py --out-dir data/moses --workers "$W" $LIMIT_ARG $F
step "3/4 RDKit oracle labels" \
  python data_pipeline/rdkit_labels.py --corpus-dir data/moses --workers "$W" $F
step "4/4 analog pair mining" \
  python data_pipeline/mine_pairs.py --corpus-dir data/moses --workers "$W" $F

echo
echo "=== done ($(date +%H:%M:%S)) ==="
python - <<'PY'
import json, pathlib
meta = pathlib.Path("data/moses/meta.json")
if meta.exists():
    m = json.loads(meta.read_text())
    print(f"corpus: {m.get('n_molecules'):,} molecules")
    print("Compare this against the other machine's meta.json. The two must match "
          "exactly; if they do not, the checkpoints trained there index a different "
          "corpus and the evaluation would pair the wrong molecules with the labels.")
PY
