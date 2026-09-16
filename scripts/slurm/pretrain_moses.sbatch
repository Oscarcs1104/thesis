#!/bin/bash
# Supervised pretraining on MOSES over exact RDKit descriptors. Everything unfrozen.
#
#   CONFIG=chemberta+gin sbatch scripts/slurm/pretrain_moses.sbatch
#   CONFIG=molformer+gin sbatch scripts/slurm/pretrain_moses.sbatch
#   CONFIG=molformer+gin LM_TOP_N=4 sbatch scripts/slurm/pretrain_moses.sbatch
#
# Both configs MUST get the same STEPS, or the comparison is about the budget rather
# than the backbones. ChemBERTa is 3 layers / ~3.3M params while MoLFormer is 12 layers
# at hidden 768, so the MoLFormer arm costs far more wall clock for the same steps --
# that is expected and is not a reason to shorten it.
#
# First run featurizes 1.94M molecules into the Hu et al. graph cache (~600 MB), once.
# mkdir -p results/logs before the first submit.

#SBATCH --job-name=thesis-pretrain
#SBATCH --account=mol_thesis
#SBATCH --partition=gpu-cvail01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=results/logs/pretrain_%j.out
#SBATCH --error=results/logs/pretrain_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p results/logs

source "${THESIS_VENV:-$HOME/venvs/thesis}/bin/activate"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

python crossmodal_model/train/pretrain_moses.py \
    --config "${CONFIG:-chemberta+gin}" \
    --max-steps "${STEPS:-40000}" \
    --batch-size "${BATCH:-256}" \
    --lm-top-n "${LM_TOP_N:-0}" \
    --num-workers 6
