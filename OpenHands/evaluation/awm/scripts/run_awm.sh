#!/usr/bin/env bash
set -eo pipefail

# AWM Online Learning Runner Script
# Usage: ./run_awm.sh [llm_config] [options...]

LLM_CONFIG=${1:-"llm.eval"}
shift || true

echo "Running AWM Online Learning with LLM config: $LLM_CONFIG"

# Default options
OUTPUT_DIR="${OUTPUT_DIR:-evaluation/evaluation_outputs/awm}"
REPO_FILTER="${REPO_FILTER:-django/django}"
INDUCTION_TRIGGER="${INDUCTION_TRIGGER:-10}"
MAX_ITERATIONS="${MAX_ITERATIONS:-100}"

poetry run python -m evaluation.awm.cli \
    --llm-config "$LLM_CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --repo-filter "$REPO_FILTER" \
    --induction-trigger "$INDUCTION_TRIGGER" \
    --max-iterations "$MAX_ITERATIONS" \
    "$@"
