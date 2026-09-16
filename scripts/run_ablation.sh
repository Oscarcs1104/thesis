#!/usr/bin/env bash
# The predictive half on a machine without SLURM: both rows, one after the other.
# The sbatch equivalent is scripts/slurm/pretrained_ablation.sbatch.
#
#   INIT=checkpoints/pretrain_moses/hybrid_graph_smiles_s2025.pt bash scripts/run_ablation.sh
#   nohup env INIT=... bash scripts/run_ablation.sh > results/logs/ablation.log 2>&1 &
#
# Both rows are run here rather than left to two invocations, because they are one
# comparison: they must see the same seeds, and under --resplit-per-seed that means the
# same partitions, which is what makes the difference testable pairwise. Running them
# separately with different seeds would throw that away silently.
#
#   INIT=<ckpt>   the MOSES-pretrained encoder. Without it only the scratch row runs,
#                 and there is nothing to compare it against.
#   SEEDS="..."   default "2025 2026 2027". More seeds is the honest way to firm up a
#                 difference; under repartitioning the spread is an order of magnitude
#                 larger than with a frozen split, so three is thin.
#   FROZEN=1      read the frozen split instead of repartitioning. Answers "is this
#                 model better than that model" with every row on identical data, but
#                 its spread covers only initialisation and its mean cannot be set
#                 against numbers measured over resampled partitions.
#   STRATEGY=...  deepchem-random (default) | random | scaffold.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/logs results

if [[ -f "${THESIS_VENV:-$HOME/venvs/thesis}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${THESIS_VENV:-$HOME/venvs/thesis}/bin/activate"
fi
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

python -c 'import importlib.util as u, sys; m=[x for x in ("rdkit","numpy","pandas","torch","torch_geometric") if not u.find_spec(x)]; sys.exit("missing: "+", ".join(m) if m else 0)'
python -c "import torch; a=torch.randn(64,64,device='cuda'); print(torch.cuda.get_device_name(0), '| kernels OK', (a@a).sum().item() is not None)"

if command -v nvidia-smi > /dev/null; then
  OTHERS=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | grep -v "^$" || true)
  [[ -n "$OTHERS" ]] && { echo "otros procesos en la GPU (aqui no hay cola):"; echo "$OTHERS" | sed 's/^/  /'; }
fi

SPLIT_FLAGS="--resplit-per-seed --split-strategy ${STRATEGY:-deepchem-random}"
TAG="resplit"
if [[ "${FROZEN:-0}" == "1" ]]; then
  SPLIT_FLAGS=""
  TAG="frozen"
fi

run_row() {
  local name="$1"; shift
  # The configs go in the filename. pretrained_ablation.py appends rather than
  # overwrites -- deliberately, so a table can be built up a row at a time -- and with a
  # name fixed per split protocol every rerun piled into the same file. Runs over 1117
  # and 1128 ESOL molecules and three different architectures ended up in one CSV, which
  # is how a "from scratch" line came to average three architectures over an n of 18.
  local cfgtag
  cfgtag=$(echo "${CONFIGS:-hybrid}" | tr ' ' '-')
  local out="results/pretrained_ablation_${TAG}_${cfgtag}_${name}.csv"
  echo
  echo "=== fila: $name -> $out ($(date +%H:%M:%S)) ==="
  python crossmodal_model/benchmark/pretrained_ablation.py \
      --datasets ${DATASETS:-esol freesolv lipo} \
      --configs ${CONFIGS:-hybrid} \
      --seeds ${SEEDS:-2025 2026 2027} \
      --out "$out" \
      $SPLIT_FLAGS "$@"
}

# Scratch first: it needs nothing, so a mistyped INIT fails before hours of work rather
# than after the first row has already been written.
if [[ -n "${INIT:-}" ]]; then
  [[ -f "$INIT" ]] || { echo "INIT=$INIT does not exist"; exit 1; }
fi

run_row scratch
if [[ -n "${INIT:-}" ]]; then
  run_row init --init-checkpoint "$INIT"
else
  echo
  echo "AVISO: sin INIT solo se corrio la fila desde cero. La comparacion que sostiene"
  echo "la mitad predictiva necesita las dos; vuelve a lanzar con INIT=<checkpoint>."
fi

echo
echo "=== done ($(date +%H:%M:%S)) ==="
python scripts/summarize_results.py
