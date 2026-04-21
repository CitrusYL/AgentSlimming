#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-MATH}"
WORKSPACE="${WORKSPACE:-workspace}"
CONFIG="${CONFIG:-config/config.yaml}"
PYTHON="${PYTHON:-python3}"
INITIAL_ROUND="${INITIAL_ROUND:-}"
TRAVERSE_SAMPLE_SIZE="${TRAVERSE_SAMPLE_SIZE:-50}"
QUANTIZE_EVAL_SAMPLES="${QUANTIZE_EVAL_SAMPLES:-}"
VALIDATION_ROUNDS="${VALIDATION_ROUNDS:-1}"
QUANTIZE_THRESHOLD="${QUANTIZE_THRESHOLD:-0.95}"
EXEC_MODEL="${EXEC_MODEL:?Set EXEC_MODEL to a model name defined in config/config.yaml}"
LOW_MODEL="${LOW_MODEL:?Set LOW_MODEL to a model name defined in config/config.yaml}"

args=(
  "$PYTHON" -m src.cli.run
  --config "$CONFIG"
  --dataset "$DATASET"
  --pipelines quantize
  --workspace "$WORKSPACE"
  --validation_rounds "$VALIDATION_ROUNDS"
  --traverse_sample_size "$TRAVERSE_SAMPLE_SIZE"
  --quantize_threshold "$QUANTIZE_THRESHOLD"
  --exec_model_name "$EXEC_MODEL"
  --quantize_low_model_name "$LOW_MODEL"
)

if [[ -n "$INITIAL_ROUND" ]]; then
  args+=(--initial_round "$INITIAL_ROUND")
fi

if [[ -n "$QUANTIZE_EVAL_SAMPLES" ]]; then
  args+=(--quantize_eval_samples "$QUANTIZE_EVAL_SAMPLES")
fi

"${args[@]}"
