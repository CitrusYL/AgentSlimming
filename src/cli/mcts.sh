#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-AIME}"
WORKSPACE="${WORKSPACE:-workspace}"
CONFIG="${CONFIG:-config/config.yaml}"
PYTHON="${PYTHON:-python3}"
INITIAL_ROUND="${INITIAL_ROUND:-1}"
MAX_ROUNDS="${MAX_ROUNDS:-10}"
VALIDATION_ROUNDS="${VALIDATION_ROUNDS:-1}"
SAMPLE="${SAMPLE:-3}"
MCTS_EVAL_SAMPLES="${MCTS_EVAL_SAMPLES:-}"
OPT_MODEL="${OPT_MODEL:?Set OPT_MODEL to a model name defined in config/config.yaml}"
EXEC_MODEL="${EXEC_MODEL:?Set EXEC_MODEL to a model name defined in config/config.yaml}"

cmd=(
  "$PYTHON" -m src.cli.run
  --config "$CONFIG"
  --dataset "$DATASET"
  --pipelines mcts
  --workspace "$WORKSPACE"
  --sample "$SAMPLE"
  --initial_round "$INITIAL_ROUND"
  --max_rounds "$MAX_ROUNDS"
  --validation_rounds "$VALIDATION_ROUNDS"
  --opt_model_name "$OPT_MODEL"
  --exec_model_name "$EXEC_MODEL"
)

if [[ -n "$MCTS_EVAL_SAMPLES" ]]; then
  cmd+=(--mcts_eval_samples "$MCTS_EVAL_SAMPLES")
fi

"${cmd[@]}"
