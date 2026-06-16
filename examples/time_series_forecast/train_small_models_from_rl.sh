#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASETS="${DATASETS:-ETTM1}"
MODELS="${MODELS:-itransformer,patchtst}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/datasets/RL}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/tmp/small-model-runs}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-3}"
DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SEED="${SEED:-42}"

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    "$PYTHON_BIN" "$PROJECT_ROOT/recipe/time_series_forecast/train_small_models_from_rl.py" \
    --datasets "$DATASETS" \
    --models "$MODELS" \
    --data-root "$DATA_ROOT" \
    --model-root "$OUTPUT_ROOT" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --device "$DEVICE" \
    --seed "$SEED" \
    "$@"
