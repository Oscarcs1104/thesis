#!/bin/bash
# Step 1 -- supervised pretraining on MOSES over exact RDKit descriptors. Unfrozen.
#
#   ARCH=hybrid                       sbatch scripts/slurm/pretrain_moses.sbatch
#   ARCH=pretrained CONFIG=chemberta+gin sbatch scripts/slurm/pretrain_moses.sbatch
#   ARCH=pretrained CONFIG=molformer+gin LM_TOP_N=4 sbatch scripts/slurm/pretrain_moses.sbatch
#
# ARCH=hybrid is the thesis's own encoder and the ONLY one whose checkpoint can go on
# to initialize the generation half -- the pretrained-backbone variants use a different
# featurization and have no character-level SMILES branch.
#
# Every arch/config MUST get the same STEPS, or the table compares budgets rather than
# encoders. MoLFormer (12 layers, hidden 768) costs far more wall clock per step than
# hybrid (~4.7M-parameter encoder); that is expected, not a reason to shorten it.
#
# HIDDEN and LAYERS set the encoder's shape, and whatever is used here propagates:
# the fine-tuning and generation scripts read these dimensions out of the checkpoint
# rather than from their own defaults, because pretrained weights only fit the shape
# they were trained in. Scaling the encoder means re-running this one step, not
# editing three files and hoping they agree.
#
# Writes checkpoints/pretrain_moses/<tag>_s<seed>.pt, e.g. hybrid_graph_smiles_s2025.pt
#
# Defaults to ARCH=hybrid, which is the only one the current plan needs. ARCH=pretrained
# with CONFIG=chemberta+gin / molformer+gin still works, but MoLFormer is 12 layers at
# hidden 768 and wants three to four hours plus --time raised; opt in deliberately.
#
# Rough cost for hybrid at 40k steps: about an hour, plus ~10 minutes to build the graph
# cache once. That is an estimate -- the script prints it/s and a live eta at every eval,
# so the first one gives you the real number. Raise it with sbatch --time=... rather than
# editing: over-asking delays the scheduler even though the job frees the GPU when done.
#
# First run featurizes 1.94M molecules into the graph cache for its schema, once.
# mkdir -p results/logs before the first submit -- SLURM opens the log files first.

# WANDB=1 logs to Weights & Biases; WANDB_MODE=offline if the node has no network.

#SBATCH --job-name=thesis-pretrain
#SBATCH --account=mol_thesis
#SBATCH --partition=gpu-cvail01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/logs/pretrain_%j.out
#SBATCH --error=results/logs/pretrain_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p results/logs

source "${THESIS_VENV:-$HOME/venvs/thesis}/bin/activate"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Weights & Biases: WANDB=1 turns it on, WANDB_MODE=offline records locally when the
# compute node has no outbound network (sync afterwards with `wandb sync wandb/offline-*`).
# WANDB_API_KEY must be exported or `wandb login` run once on the login node.
WANDB_FLAGS=""
if [[ "${WANDB:-0}" == "1" ]]; then
  WANDB_FLAGS="--use-wandb"
  [[ -n "${WANDB_GROUP:-}" ]] && WANDB_FLAGS="$WANDB_FLAGS --wandb-group $WANDB_GROUP"
  [[ -n "${WANDB_ENTITY:-}" ]] && WANDB_FLAGS="$WANDB_FLAGS --wandb-entity $WANDB_ENTITY"
fi

ARCH="${ARCH:-hybrid}"
ARM_FLAGS=""
case "${ARM:-graph+smiles}" in
  graph+smiles) ARM_FLAGS="" ;;
  graph-only)   ARM_FLAGS="--no-use-smiles" ;;
  smiles-only)  ARM_FLAGS="--no-use-graph" ;;
  *) echo "ARM must be graph+smiles, graph-only or smiles-only"; exit 1 ;;
esac

python crossmodal_model/train/pretrain_moses.py \
    --arch "$ARCH" \
    --config "${CONFIG:-chemberta+gin}" \
    --hidden-dim "${HIDDEN:-256}" \
    --num-layers "${LAYERS:-3}" \
    --max-steps "${STEPS:-40000}" \
    --batch-size "${BATCH:-256}" \
    --lm-top-n "${LM_TOP_N:-0}" \
    --seed "${SEED:-2025}" \
    --num-workers 6 \
    $ARM_FLAGS \
    $WANDB_FLAGS
