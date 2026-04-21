#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${EXEC_MODEL:?Set EXEC_MODEL to a model name defined in config/config.yaml}"
: "${LOW_MODEL:?Set LOW_MODEL to a model name defined in config/config.yaml}"
export TRAVERSE_SAMPLE_SIZE="${TRAVERSE_SAMPLE_SIZE:-32}"
export QUANTIZE_EVAL_SAMPLES="${QUANTIZE_EVAL_SAMPLES:-140}"

IFS=' ' read -r -a DATASETS <<< "${DATASETS:-MBPP}"

for dataset in "${DATASETS[@]}"; do
  echo "Running quantization: dataset=${dataset}, exec_model=${EXEC_MODEL}, low_model=${LOW_MODEL}"
  DATASET="$dataset" "$SCRIPT_DIR/quantize.sh"
done
