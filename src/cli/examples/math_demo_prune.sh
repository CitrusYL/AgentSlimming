#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATASET="${DATASET:-MATHDEMO}"
export WORKSPACE="${WORKSPACE:-workspace}"
export INITIAL_ROUND="${INITIAL_ROUND:-2}"
export TRAVERSE_SAMPLE_SIZE="${TRAVERSE_SAMPLE_SIZE:-8}"
export PRUNE_EVAL_SAMPLES="${PRUNE_EVAL_SAMPLES:-8}"
export VALIDATION_ROUNDS="${VALIDATION_ROUNDS:-1}"
export PRUNE_THRESHOLD="${PRUNE_THRESHOLD:-0.95}"

"$SCRIPT_DIR/../prune.sh"
