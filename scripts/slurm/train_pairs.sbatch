#!/bin/bash
# Step 3 -- conditional generation: (M_a, delta property) -> M_b. One ablation arm per job.
#
#   ARM=graph+smiles INIT=checkpoints/pretrain_moses/hybrid_graph_smiles_s2025.pt \
#       sbatch scripts/slurm/train_pairs.sbatch
#   ARM=graph-only   sbatch scripts/slurm/train_pairs.sbatch
#   ARM=smiles-only  sbatch scripts/slurm/train_pairs.sbatch
#
# INIT is the MOSES-pretrained HybridMoLA encoder from step 1. It must have been
# pretrained with the SAME ARM: loading the graph-only checkpoint into a graph+smiles
# run would leave the SMILES branch uninitialized while the log says "encoder
# initialized", so the script refuses it. Pretrain each arm you intend to run.
#
# The three arms MUST use the same STEPS. They run sequentially on one GPU, and an arm
# that trained longer because its epochs were cheaper would make the comparison about
# wall clock rather than about the modalities.
#
# The first run featurizes 1.94M molecules into the OGB graph cache (~1.4 GB, written to
# data/moses/). Later runs and the other arms load it in seconds.
#
# mkdir -p results/logs before the first submit -- SLURM opens the log files first.

#SBATCH --job-name=thesis-pairs
#SBATCH --account=mol_thesis
#SBATCH --partition=gpu-cvail01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=results/logs/pairs_%j.out
#SBATCH --error=results/logs/pairs_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p results/logs

source "${THESIS_VENV:-$HOME/venvs/thesis}/bin/activate"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

case "${ARM:-graph+smiles}" in
  graph+smiles) FLAGS="" ;;
  graph-only)   FLAGS="--no-use-smiles" ;;
  smiles-only)  FLAGS="--no-use-graph" ;;
  *) echo "ARM must be graph+smiles, graph-only or smiles-only"; exit 1 ;;
esac

if [[ -n "${INIT:-}" ]]; then
  [[ -f "$INIT" ]] || { echo "INIT=$INIT does not exist"; exit 1; }
  FLAGS="$FLAGS --init-encoder $INIT"
fi
# FREEZE=1 keeps the pretrained encoder fixed. It weakens the claim from "the fused
# encoder makes the decoder work" to "a frozen pretrained encoder helps", so it is off
# unless asked for.
[[ "${FREEZE:-0}" == "1" ]] && FLAGS="$FLAGS --freeze-encoder"

python crossmodal_model/generation/train_pairs.py \
    --corpus-dir data/moses \
    --max-steps "${STEPS:-60000}" \
    --batch-size "${BATCH:-256}" \
    --seed "${SEED:-2025}" \
    --num-workers 6 \
    $FLAGS
