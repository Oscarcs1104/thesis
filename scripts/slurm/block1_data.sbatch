#!/bin/bash
# Block 1 -- MOSES corpus + oracle labels + analog pairs. CPU only, no GPU.
#
#   LIMIT=5000  sbatch scripts/slurm/block1_data.sbatch   # plumbing check, ~1 min
#   LIMIT=50000 sbatch scripts/slurm/block1_data.sbatch   # smallest useful smoke, ~5 min
#   FORCE=1     sbatch scripts/slurm/block1_data.sbatch   # the full 1.9M
#
# Below ~50k, stage 1c finds almost no pairs, and that is arithmetic rather than a
# bug: MOSES holds hundreds of thousands of distinct Murcko scaffolds, so in a
# small sample nearly every molecule is alone in its scaffold and has nothing to
# pair with. A LIMIT=5000 run still exercises 1a and 1b end to end.
#
# FIRST TIME ONLY: `mkdir -p results/logs` before submitting. SLURM opens the
# --output/--error files before the script runs, so the mkdir below is too late
# and the job dies at launch with no usable message.
#
# Override resources at submit time rather than editing this file:
#
#   sbatch --cpus-per-task=16 --mem=24G scripts/slurm/block1_data.sbatch
#
# CPUs only change how long it takes. Memory is the one that decides whether it
# runs at all: ~2G for the smoke, ~10-14G for the full 1.9M (the peak is holding
# every canonical SMILES + SELFIES string in memory at once, before encoding).

#SBATCH --job-name=thesis-block1
#SBATCH --account=mol_thesis
#SBATCH --partition=cpu-cvail01
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=results/logs/block1_%j.out
#SBATCH --error=results/logs/block1_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p results/logs

source "${THESIS_VENV:-$HOME/venvs/thesis}/bin/activate"

# Each stage already forks one process per core; BLAS threads on top of that
# oversubscribe the node and make it slower.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1

W="${SLURM_CPUS_PER_TASK:-8}"
LIMIT_ARG=${LIMIT:+--limit $LIMIT}
F=${FORCE:+--force}

# Fail in seconds on a bad env instead of after the corpus download.
python -c "import rdkit, selfies, numpy, pandas"

python data_pipeline/moses.py        --out-dir data/moses    --workers "$W" $LIMIT_ARG $F
python data_pipeline/rdkit_labels.py --corpus-dir data/moses --workers "$W" $F
# No --balance-delta on the first run: read the delta histogram this prints first.
python data_pipeline/mine_pairs.py   --corpus-dir data/moses --workers "$W" $F
