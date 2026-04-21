#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-MATH}"
ROUND="${ROUND:-1}"
WORKSPACE="${WORKSPACE:-workspace}"
CONFIG="${CONFIG:-config/config.yaml}"
PYTHON="${PYTHON:-python3}"
DATA_PATH="${DATA_PATH:-data/datasets/math_test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-test_results}"
GRAPH="${GRAPH:-${WORKSPACE}.${DATASET}.workflows.round_${ROUND}.graph}"
MODEL="${MODEL:?Set MODEL to a model name defined in config/config.yaml}"

cmd=(
  "$PYTHON" -m src.cli.test_graph
  --config "$CONFIG"
  --graph "$GRAPH"
  --dataset "$DATASET"
  --data_path "$DATA_PATH"
  --model "$MODEL"
  --output_dir "$OUTPUT_DIR"
)

"${cmd[@]}"
