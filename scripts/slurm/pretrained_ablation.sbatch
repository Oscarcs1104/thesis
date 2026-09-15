#!/bin/bash
# Property-prediction ablation over pretrained backbones. Needs a GPU.
#
#   sbatch scripts/slurm/pretrained_ablation.sbatch
#   sbatch --export=CONFIGS="gin chemberta+gin" scripts/slurm/pretrained_ablation.sbatch
#
# Frozen backbones are precomputed once per split, so the 100 epochs afterwards are a
# tiny fusion module over cached tensors. Expect roughly a minute per cell, ~45 minutes
# for the full 5 configs x 3 datasets x 3 seeds. The time below is a ceiling, not a
# reservation -- the job frees the GPU the moment it finishes -- but asking for more
# than you need makes the scheduler queue you later, so keep it honest.
#
# Time one cell before committing to the grid:
#   sbatch --export=DATASETS=esol,CONFIGS=gin,SEEDS=2025 scripts/slurm/pretrained_ablation.sbatch
#
# First run downloads the GIN checkpoint (7.5 MB) and the HF backbones (~1 GB).
# mkdir -p results/logs before the first submit -- SLURM opens the log files before
# the script runs.

#SBATCH --job-name=thesis-pretrain-abl
#SBATCH --account=mol_thesis
#SBATCH --partition=gpu-cvail01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=results/logs/pretrained_abl_%j.out
#SBATCH --error=results/logs/pretrained_abl_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p results/logs

source "${THESIS_VENV:-$HOME/venvs/thesis}/bin/activate"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false   # the DataLoader already forks; HF would warn per batch

python -c "import torch; print('GPU:', torch.cuda.get_device_name(0), '| VRAM %.0f GB' % (torch.cuda.get_device_properties(0).total_memory/1e9))"

python crossmodal_model/benchmark/pretrained_ablation.py \
    --datasets ${DATASETS:-esol freesolv lipo} \
    --configs ${CONFIGS:-chemberta molformer gin chemberta+gin molformer+gin} \
    --seeds ${SEEDS:-2025 2026 2027} \
    --out results/pretrained_ablation.csv
