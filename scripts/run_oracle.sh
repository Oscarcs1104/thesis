#!/usr/bin/env bash
# Run the oracle evaluation over every pair checkpoint, on a machine without SLURM.
#
#   bash scripts/run_oracle.sh                 # every checkpoint, full sweep
#   SMOKE=1 bash scripts/run_oracle.sh         # 20 seeds, 3 bins, two minutes
#   bash scripts/run_oracle.sh checkpoints/pairs/pairs_graph_smiles_scratch_s2025.pt
#
# In the background, surviving a closed terminal:
#
#   nohup bash scripts/run_oracle.sh > results/logs/oracle.log 2>&1 &
#   tail -f results/logs/oracle.log
#
# Sequential on purpose. There is one GPU, and four of these at once would compete
# for VRAM and finish later than they would in a queue -- or fail on the fourth.
# `wait` is not used for the same reason: each run must finish before the next starts.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/logs results/oracle

if [[ -f "${THESIS_VENV:-$HOME/venvs/thesis}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${THESIS_VENV:-$HOME/venvs/thesis}/bin/activate"
fi
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# A wheel without sm_120 imports fine and reports cuda available, then dies on the
# first kernel. Ten seconds here beats discovering it after the corpus has loaded.
python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("no CUDA device visible")
print(f"{torch.cuda.get_device_name(0)} | torch {torch.__version__}")
(torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda")).sum().item()
print("GPU kernels OK")
PY

if [[ $# -gt 0 ]]; then
  CKPTS=("$@")
else
  shopt -s nullglob
  CKPTS=(checkpoints/pairs/*.pt)
  shopt -u nullglob
fi
[[ ${#CKPTS[@]} -gt 0 ]] || { echo "no checkpoints in checkpoints/pairs/"; exit 1; }

if [[ "${SMOKE:-0}" == "1" ]]; then
  EXTRA=(--n-seeds 20 --bins 0 10 19 --guidance 1.0)
else
  EXTRA=(--n-seeds "${SEEDS:-200}")
  read -r -a GUID <<< "${GUIDANCE:-1.0 2.0 3.0}"
  EXTRA+=(--guidance "${GUID[@]}")
fi

echo "== ${#CKPTS[@]} checkpoint(s), one at a time =="
FAILED=()
for CKPT in "${CKPTS[@]}"; do
  NAME=$(basename "$CKPT" .pt)
  LOG="results/logs/oracle_${NAME}.log"
  echo
  echo "--- $NAME -> $LOG ---"
  # One failure must not abandon the rest: three good arms are still a figure, and
  # the summary at the end says which one is missing rather than leaving it to the
  # scrollback. set -e would otherwise take the whole script down here.
  if python crossmodal_model/generation/eval_oracle.py \
        --ckpt "$CKPT" \
        --corpus-dir data/moses \
        --property "${PROPERTY:-logp}" \
        --seed "${SEED:-2025}" \
        --batch-size "${BATCH:-128}" \
        --num-workers "${WORKERS:-6}" \
        "${EXTRA[@]}" > "$LOG" 2>&1; then
    tail -n 12 "$LOG"
  else
    echo "FAILED -- last lines of $LOG:"
    tail -n 20 "$LOG"
    FAILED+=("$NAME")
  fi
done

echo
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "== ${#FAILED[@]} failed: ${FAILED[*]} =="
else
  echo "== all ${#CKPTS[@]} finished =="
fi
python scripts/summarize_results.py
[[ ${#FAILED[@]} -eq 0 ]]
