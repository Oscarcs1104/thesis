#!/bin/bash
# Step 2 -- fine-tune on ESOL / FreeSolv / Lipophilicity. Property prediction.
#
#   sbatch scripts/slurm/pretrained_ablation.sbatch                       # everything, from scratch
#   CONFIGS=hybrid sbatch scripts/slurm/pretrained_ablation.sbatch        # hybrid, no pretraining
#   CONFIGS=hybrid INIT=checkpoints/pretrain_moses/hybrid_graph_smiles_s2025.pt \
#       sbatch scripts/slurm/pretrained_ablation.sbatch                   # hybrid, pretrained
#
# The CSV carries a `pretrained` column, so the with/without rows land in the same file
# and compare directly. Run the same CONFIGS both ways.
#
# INIT must match CONFIGS: a checkpoint pretrained as one configuration is refused for
# another rather than loaded partially.
#
# Frozen backbones are precomputed once per split, so the epochs afterwards are a small
# fusion module over cached tensors -- about a minute per cell. `hybrid` trains end to
# end and has nothing to cache, so it is slower per cell.
#
# Time one cell before committing to the grid:
#   DATASETS=esol CONFIGS=gin SEEDS=2025 sbatch scripts/slurm/pretrained_ablation.sbatch
#
# First run downloads the GIN checkpoint (7.5 MB) and the HF backbones (~1 GB).
# mkdir -p results/logs before the first submit -- SLURM opens the log files first.

#SBATCH --job-name=thesis-finetune
#SBATCH --account=mol_thesis
#SBATCH --partition=gpu-cvail01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=results/logs/finetune_%j.out
#SBATCH --error=results/logs/finetune_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p results/logs

source "${THESIS_VENV:-$HOME/venvs/thesis}/bin/activate"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

python -c "import torch; print('GPU:', torch.cuda.get_device_name(0), '| VRAM %.0f GB' % (torch.cuda.get_device_properties(0).total_memory/1e9))"

INIT_FLAG=""
if [[ -n "${INIT:-}" ]]; then
  [[ -f "$INIT" ]] || { echo "INIT=$INIT does not exist"; exit 1; }
  INIT_FLAG="--init-checkpoint $INIT"
fi

python crossmodal_model/benchmark/pretrained_ablation.py \
    --datasets ${DATASETS:-esol freesolv lipo} \
    --configs ${CONFIGS:-chemberta molformer gin chemberta+gin molformer+gin hybrid} \
    --seeds ${SEEDS:-2025 2026 2027} \
    --out "${OUT:-results/pretrained_ablation.csv}" \
    $INIT_FLAG
