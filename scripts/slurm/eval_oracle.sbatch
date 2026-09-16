#!/bin/bash
# Step 4 -- the deliverable: ask for a property change, measure what came out with RDKit.
#
# One job per checkpoint. The default sweeps every bin of logP at three guidance weights
# over 200 test-split seeds, which is ~12,600 generations.
#
#   CKPT=checkpoints/pairs/pairs_graph_smiles_scratch_s2025.pt   sbatch scripts/slurm/eval_oracle.sbatch
#   CKPT=checkpoints/pairs/pairs_graph-only_scratch_s2025.pt     sbatch scripts/slurm/eval_oracle.sbatch
#   CKPT=checkpoints/pairs/pairs_smiles-only_scratch_s2025.pt    sbatch scripts/slurm/eval_oracle.sbatch
#   CKPT=checkpoints/pairs/pairs_graph_smiles_pretrained_s2025.pt sbatch scripts/slurm/eval_oracle.sbatch
#
# SEED must match the training run's, or the "test" seeds were in that run's training
# set and every number here is measured on molecules the model memorised.
#
# PROPERTY=logp|tpsa|qed|mw    which one is conditioned on; the rest are set to null.
# SEEDS=200                    seed molecules, each asked for every bin.
# GUIDANCE="1.0 2.0 3.0"       classifier-free guidance sweep. If the curve does not
#                              steepen with w, the condition is barely being used --
#                              that is the diagnostic, not a tuning knob.
# SMOKE=1                      20 seeds, 3 bins, w=1. Two minutes, proves it runs.
#
# Reading the output: `spearman(requested bin, obtained delta)` is the number. It is
# computed within-seed -- every bin is asked of the same molecules -- so unlike the old
# evaluation it cannot be inflated by the seed's own property value. Read it next to the
# copy rate: a model that returns the seed unchanged has a constant delta and scores
# zero here, which is the honest answer, but only the copy rate says why.

#SBATCH --job-name=thesis-oracle
#SBATCH --account=mol_thesis
#SBATCH --partition=gpu-cvail01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/logs/oracle_%j.out
#SBATCH --error=results/logs/oracle_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p results/logs

source "${THESIS_VENV:-$HOME/venvs/thesis}/bin/activate"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

: "${CKPT:?set CKPT to the checkpoint to evaluate}"
[[ -f "$CKPT" ]] || { echo "CKPT=$CKPT does not exist"; exit 1; }

if [[ "${SMOKE:-0}" == "1" ]]; then
  EXTRA="--n-seeds 20 --bins 0 10 19 --guidance 1.0"
else
  EXTRA="--n-seeds ${SEEDS:-200} --guidance ${GUIDANCE:-1.0 2.0 3.0}"
fi

python crossmodal_model/generation/eval_oracle.py \
    --ckpt "$CKPT" \
    --corpus-dir data/moses \
    --property "${PROPERTY:-logp}" \
    --seed "${SEED:-2025}" \
    --batch-size "${BATCH:-128}" \
    --num-workers 6 \
    $EXTRA
