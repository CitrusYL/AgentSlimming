#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${EXEC_MODEL:?Set EXEC_MODEL to a model name defined in config/config.yaml}"
export TRAVERSE_SAMPLE_SIZE="${TRAVERSE_SAMPLE_SIZE:-32}"
export PRUNE_EVAL_SAMPLES="${PRUNE_EVAL_SAMPLES:-140}"

IFS=' ' read -r -a DATASETS <<< "${DATASETS:-AIME MBPP}"

for dataset in "${DATASETS[@]}"; do
  echo "Running pruning: dataset=${dataset}, exec_model=${EXEC_MODEL}"
  DATASET="$dataset" "$SCRIPT_DIR/prune.sh"
done
