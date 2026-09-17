#!/usr/bin/env bash
# Pretrain the encoder on a machine without SLURM. The sbatch equivalent is
# scripts/slurm/pretrain_moses.sbatch.
#
#   CORPUS_TAG=qm9 bash scripts/run_pretrain.sh
#   nohup env CORPUS_TAG=qm9 bash scripts/run_pretrain.sh > results/logs/pretrain_qm9.log 2>&1 &
#
#   CORPUS_TAG=...  goes into the checkpoint name. Leave it empty only when rebuilding the
#                   run that already owns the untagged name: two encoders pretrained on
#                   different corpora resolve to the same file otherwise, and the second
#                   replaces the first without a word.
#   STEPS=40000     the budget. Fixed in steps rather than epochs, so widening the corpus
#                   costs no extra time and changes how often each molecule is seen.
#   ARM=graph+smiles | graph-only | smiles-only
#
# About 12 minutes for 40k steps over 1.94M molecules on an RTX PRO 6000.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/logs checkpoints/pretrain_moses

if [[ -f "${THESIS_VENV:-$HOME/venvs/thesis}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${THESIS_VENV:-$HOME/venvs/thesis}/bin/activate"
fi
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

# A wheel without sm_120 imports fine and reports cuda available, then dies on the first
# kernel. Ten seconds here beats finding out after the corpus has loaded.
python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("no CUDA device visible")
print(f"{torch.cuda.get_device_name(0)} | torch {torch.__version__}")
(torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda")).sum().item()
print("GPU kernels OK")
PY

# There is no scheduler here to queue behind. VRAM free is not the same as GPU free: a
# card at 100% utilisation with 25 GB spare will still halve both jobs.
if command -v nvidia-smi > /dev/null; then
  echo "--- GPU ahora mismo ---"
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true
  OTHERS=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | grep -v "^$" || true)
  if [[ -n "$OTHERS" ]]; then
    echo "procesos en la GPU:"; echo "$OTHERS" | sed 's/^/  /'
    echo "  (si la utilizacion esta alta y no es tuyo, compartir os frena a los dos)"
  else
    echo "  libre"
  fi
  echo "-----------------------"
fi

case "${ARM:-graph+smiles}" in
  graph+smiles) ARM_FLAGS="" ;;
  graph-only)   ARM_FLAGS="--no-use-smiles" ;;
  smiles-only)  ARM_FLAGS="--no-use-graph" ;;
  *) echo "ARM must be graph+smiles, graph-only or smiles-only"; exit 1 ;;
esac

WANDB_FLAGS=""
if [[ "${WANDB:-0}" == "1" ]]; then
  WANDB_FLAGS="--use-wandb"
  [[ -n "${WANDB_GROUP:-}" ]] && WANDB_FLAGS="$WANDB_FLAGS --wandb-group $WANDB_GROUP"
fi

python crossmodal_model/train/pretrain_moses.py \
    --arch "${ARCH:-hybrid}" \
    --corpus-dir "${CORPUS_DIR:-data/moses}" \
    --hidden-dim "${HIDDEN:-256}" \
    --num-layers "${LAYERS:-3}" \
    --max-steps "${STEPS:-40000}" \
    --batch-size "${BATCH:-256}" \
    --seed "${SEED:-2025}" \
    --corpus-tag "${CORPUS_TAG:-}" \
    --num-workers "${WORKERS:-6}" \
    $ARM_FLAGS \
    $WANDB_FLAGS

echo
ls -lh checkpoints/pretrain_moses/
