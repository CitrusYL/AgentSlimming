#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-MATH}"
WORKSPACE="${WORKSPACE:-workspace}"
CONFIG="${CONFIG:-config/config.yaml}"
PYTHON="${PYTHON:-python3}"
MAX_ROUNDS="${MAX_ROUNDS:-5}"
VALIDATION_ROUNDS="${VALIDATION_ROUNDS:-1}"
SAMPLE="${SAMPLE:-3}"
FINETUNE_EVAL_SAMPLES="${FINETUNE_EVAL_SAMPLES:-}"
OPT_MODEL="${OPT_MODEL:?Please set the OPT_MODEL environment variable to specify the model for fine-tuning.}"
EXEC_MODEL="${EXEC_MODEL:?Please set the EXEC_MODEL environment variable to specify the model for execution.}"

args=(
	"$PYTHON" -m src.cli.run
	--config "$CONFIG"
	--dataset "$DATASET"
	--pipelines finetune
	--workspace "$WORKSPACE"
	--sample "$SAMPLE"
	--max_rounds "$MAX_ROUNDS"
	--validation_rounds "$VALIDATION_ROUNDS"
	--opt_model_name "$OPT_MODEL"
	--exec_model_name "$EXEC_MODEL"
)

if [[ -n "$FINETUNE_EVAL_SAMPLES" ]]; then
	args+=(--finetune_eval_samples "$FINETUNE_EVAL_SAMPLES")
fi

"${args[@]}"
