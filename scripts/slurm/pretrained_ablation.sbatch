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
# hybrid trains end to end, so nothing is cached: roughly a few minutes per cell, well
# under an hour for 3 datasets x 3 seeds. The pretrained-backbone rows instead
# precompute their frozen features once per split and take about a minute per cell.
#
# Time one cell before committing to the grid:
#   DATASETS=esol SEEDS=2025 sbatch scripts/slurm/pretrained_ablation.sbatch
# mkdir -p results/logs before the first submit -- SLURM opens the log files first.

# WANDB=1 logs to Weights & Biases; WANDB_MODE=offline if the node has no network.

#SBATCH --job-name=thesis-finetune
#SBATCH --account=mol_thesis
#SBATCH --partition=gpu-cvail01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=results/logs/finetune_%j.out
#SBATCH --error=results/logs/finetune_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p results/logs

source "${THESIS_VENV:-$HOME/venvs/thesis}/bin/activate"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

python -c "import torch; print('GPU:', torch.cuda.get_device_name(0), '| VRAM %.0f GB' % (torch.cuda.get_device_properties(0).total_memory/1e9))"

# Weights & Biases: WANDB=1 turns it on, WANDB_MODE=offline records locally when the
# compute node has no outbound network (sync afterwards with `wandb sync wandb/offline-*`).
# WANDB_API_KEY must be exported or `wandb login` run once on the login node.
WANDB_FLAGS=""
if [[ "${WANDB:-0}" == "1" ]]; then
  WANDB_FLAGS="--use-wandb"
  [[ -n "${WANDB_GROUP:-}" ]] && WANDB_FLAGS="$WANDB_FLAGS --wandb-group $WANDB_GROUP"
  [[ -n "${WANDB_ENTITY:-}" ]] && WANDB_FLAGS="$WANDB_FLAGS --wandb-entity $WANDB_ENTITY"
fi

INIT_FLAG=""
if [[ -n "${INIT:-}" ]]; then
  [[ -f "$INIT" ]] || { echo "INIT=$INIT does not exist"; exit 1; }
  INIT_FLAG="--init-checkpoint $INIT"
fi

python crossmodal_model/benchmark/pretrained_ablation.py \
    --datasets ${DATASETS:-esol freesolv lipo} \
    --configs ${CONFIGS:-hybrid} \
    --seeds ${SEEDS:-2025 2026 2027} \
    --out "${OUT:-results/pretrained_ablation.csv}" \
    $INIT_FLAG \
    $WANDB_FLAGS
