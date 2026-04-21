#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-MATH}"
WORKSPACE="${WORKSPACE:-workspace}"
CONFIG="${CONFIG:-config/config.yaml}"
PYTHON="${PYTHON:-python3}"
INITIAL_ROUND="${INITIAL_ROUND:-}"
TRAVERSE_SAMPLE_SIZE="${TRAVERSE_SAMPLE_SIZE:-50}"
PRUNE_EVAL_SAMPLES="${PRUNE_EVAL_SAMPLES:-}"
VALIDATION_ROUNDS="${VALIDATION_ROUNDS:-1}"
PRUNE_THRESHOLD="${PRUNE_THRESHOLD:-0.95}"
EXEC_MODEL="${EXEC_MODEL:?Set EXEC_MODEL to a model name defined in config/config.yaml}"

args=(
  "$PYTHON" -m src.cli.run
  --config "$CONFIG"
  --dataset "$DATASET"
  --pipelines prune
  --workspace "$WORKSPACE"
  --traverse_sample_size "$TRAVERSE_SAMPLE_SIZE"
  --validation_rounds "$VALIDATION_ROUNDS"
  --prune_threshold "$PRUNE_THRESHOLD"
  --exec_model_name "$EXEC_MODEL"
)

if [[ -n "$INITIAL_ROUND" ]]; then
  args+=(--initial_round "$INITIAL_ROUND")
fi

if [[ -n "$PRUNE_EVAL_SAMPLES" ]]; then
  args+=(--prune_eval_samples "$PRUNE_EVAL_SAMPLES")
fi

"${args[@]}"
