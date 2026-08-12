#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:-multimodal_model}"
DATA_PATH="${2:-data/esol.csv,data/freesolv.csv,data/lipo.csv}"
DEVICE="${3:-cuda}"

powershell -NoProfile -ExecutionPolicy Bypass -File "run_experiment.ps1" -RunName "$RUN_NAME" -DataPath "$DATA_PATH" -Device "$DEVICE" -TrainingMode predictor
