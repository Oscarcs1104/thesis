#!/bin/bash
# =============================================================================
# Block 1 -- build the MOSES pretraining corpus, oracle labels and analog pairs.
#
# CPU ONLY. Every stage is RDKit + multiprocessing; nothing touches a GPU. Do not
# request one -- it would sit idle for the whole job and block the training queue.
#
# Submit:
#   sbatch scripts/slurm/block1_data.sbatch                 # full ~1.9M run
#   SMOKE=1 sbatch scripts/slurm/block1_data.sbatch         # 50k molecules, ~5 min, do this first
#
# Watch:
#   squeue -u $USER
#   tail -f results/logs/block1_<jobid>.out
# =============================================================================

# ----- CLUSTER-SPECIFIC: check these three against `sinfo` before submitting ---
#SBATCH --partition=cpu
###SBATCH --account=YOUR_ACCOUNT          # uncomment and set if your cluster needs one
# -----------------------------------------------------------------------------

#SBATCH --job-name=thesis-block1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=results/logs/block1_%j.out
#SBATCH --error=results/logs/block1_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p results/logs

# ----- environment -----------------------------------------------------------
# Works with either a venv or a conda env, because a shared /opt miniforge often
# has no writable pkgs dir for a normal user and `conda create` just fails there.
#   venv  : THESIS_VENV=~/venvs/thesis sbatch ...   (or leave the default below)
#   conda : THESIS_CONDA_ENV=thesis    sbatch ...
VENV_PATH="${THESIS_VENV:-$HOME/venvs/thesis}"
if [[ -f "$VENV_PATH/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$VENV_PATH/bin/activate"
    echo "env        : venv $VENV_PATH"
elif [[ -n "${THESIS_CONDA_ENV:-}" ]] && command -v conda >/dev/null 2>&1; then
    # conda activate needs the shell hook: sbatch runs a non-interactive shell
    # that never sources the user profile, so a bare `conda activate` fails.
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$THESIS_CONDA_ENV"
    echo "env        : conda $THESIS_CONDA_ENV"
else
    echo "env        : none activated -- using whatever python is on PATH"
fi

python -c "import rdkit, selfies, numpy, pandas" || {
    echo "FATAL: the environment is missing rdkit/selfies/numpy/pandas. Activate the right"
    echo "       one (THESIS_VENV=... or THESIS_CONDA_ENV=...) and resubmit."
    exit 1
}

# Every stage already parallelizes with multiprocessing over CPUS. Letting BLAS
# spawn its own threads inside each worker oversubscribes the node and makes the
# whole thing slower, not faster.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

WORKERS="${SLURM_CPUS_PER_TASK:-8}"
CORPUS_DIR="data/moses"

if [[ "${SMOKE:-0}" == "1" ]]; then
    LIMIT_ARG="--limit 50000"
    echo "### SMOKE RUN: 50k molecules ###"
else
    LIMIT_ARG=""
fi

echo "=========================================================="
echo "job        : ${SLURM_JOB_ID:-local} on $(hostname)"
echo "workers    : $WORKERS"
echo "corpus dir : $CORPUS_DIR"
echo "commit     : $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "started    : $(date)"
echo "=========================================================="

# ----- 1a: corpus ------------------------------------------------------------
echo ""; echo "### 1a  corpus: SMILES -> canonical -> SELFIES -> int16 ###"
python data_pipeline/moses.py \
    --out-dir "$CORPUS_DIR" \
    --workers "$WORKERS" \
    $LIMIT_ARG \
    ${FORCE:+--force}

# ----- 1b: oracle labels -----------------------------------------------------
echo ""; echo "### 1b  labels: logP / TPSA / QED / MW ###"
python data_pipeline/rdkit_labels.py \
    --corpus-dir "$CORPUS_DIR" \
    --workers "$WORKERS" \
    ${FORCE:+--force}

# ----- 1c: analog pairs ------------------------------------------------------
# Deliberately NO --balance-delta on the first run: the balancing cap is strongly
# non-linear in how peaked the deltas turn out to be, so read the histogram this
# prints (and pairs_meta.json) before deciding whether to enable it.
echo ""; echo "### 1c  pairs: scaffold buckets -> Tanimoto band -> deltas ###"
python data_pipeline/mine_pairs.py \
    --corpus-dir "$CORPUS_DIR" \
    --workers "$WORKERS" \
    ${FORCE:+--force}

echo ""
echo "=========================================================="
echo "finished   : $(date)"
echo "----------------------------------------------------------"
python - <<'PY'
import json
from pathlib import Path
d = Path("data/moses")
meta = json.loads((d / "meta.json").read_text())
dedup = json.loads((d / "dedup_report.json").read_text())
pairs = json.loads((d / "pairs_meta.json").read_text())
print(f"molecules      : {meta['n_molecules']}   (rejected {meta['n_rejected']}, "
      f"dupes {meta['n_internal_duplicates']}, eval-overlap {meta['n_removed_eval_overlap']})")
print(f"max_len/vocab  : {meta['max_len']} tokens / {meta['vocab_size']} types")
so = dedup["scaffold_overlap"]
print(f"scaffold overlap with test sets: {so['n_shared_scaffolds']} scaffolds, "
      f"{so['fraction_of_corpus']:.1%} of corpus (reported, not removed)")
print(f"pairs          : {pairs['n_pairs']}   ({pairs['n_identity']} identity)")
lp = pairs["delta_stats"]["logp"]
print(f"delta logP     : std={lp['std']:.2f}  p1={lp['p1']:.2f}  p99={lp['p99']:.2f}  "
      f"|d|>1 = {lp['frac_abs_gt_1']:.1%}   <-- THE number to check")
if lp["frac_abs_gt_1"] < 0.05:
    print("  [!] almost every pair is a near-copy: the conditioning has little to learn.")
    print("      Lower --tanimoto-max, or re-run mine_pairs.py with --balance-delta logp.")
PY
echo "=========================================================="
