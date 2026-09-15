#!/bin/bash
# Block 1 -- MOSES corpus + oracle labels + analog pairs. CPU only, no GPU.
#
#   SMOKE=1 sbatch scripts/slurm/block1_data.sbatch    # 50k molecules, ~5 min
#   FORCE=1 sbatch scripts/slurm/block1_data.sbatch    # the full 1.9M
#
# Check --partition against `sinfo` before the first submit.

#SBATCH --job-name=thesis-block1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
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
LIMIT=$([[ "${SMOKE:-0}" == "1" ]] && echo "--limit 50000" || echo "")
F=${FORCE:+--force}

# Fail in seconds on a bad env instead of after the corpus download.
python -c "import rdkit, selfies, numpy, pandas"

python data_pipeline/moses.py        --out-dir data/moses    --workers "$W" $LIMIT $F
python data_pipeline/rdkit_labels.py --corpus-dir data/moses --workers "$W" $F
# No --balance-delta on the first run: read the delta histogram this prints first.
python data_pipeline/mine_pairs.py   --corpus-dir data/moses --workers "$W" $F
