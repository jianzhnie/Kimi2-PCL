#!/bin/bash
# Compare generated weight map (from model definition) vs actual checkpoint weight map.
#
# Usage:
#   bash scripts/compare_weight_maps.sh
#   GENERATED=weight_map.json ACTUAL=model_param_hf.json bash scripts/compare_weight_maps.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

GENERATED="${GENERATED:-${PROJECT_DIR}/model_param_hf_generated.json}"
ACTUAL="${ACTUAL:-${PROJECT_DIR}/model_param_hf.json}"

python3 "${PROJECT_DIR}/utils/compare_weight_maps.py" "$GENERATED" "$ACTUAL"
