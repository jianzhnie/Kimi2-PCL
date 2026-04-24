#!/bin/bash
# Generate weight map JSON from model definition (no GPU/memory required).
#
# Usage:
#   bash scripts/get_hf_weights_from_model.sh
#   CONFIG=models/config.json OUTPUT=weight_map.json bash scripts/get_hf_weights_from_model.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

CONFIG="${CONFIG:-${PROJECT_DIR}/models/config.json}"
OUTPUT="${OUTPUT:-${PROJECT_DIR}/model_param_hf_generated.json}"
DTYPE="${DTYPE:-bfloat16}"

python "${PROJECT_DIR}/utils/get_hf_weights_from_model.py" \
    --config "$CONFIG" \
    --output "$OUTPUT" \
    --dtype "$DTYPE" \
    --pretty
