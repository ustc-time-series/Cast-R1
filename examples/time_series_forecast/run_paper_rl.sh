#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CAST_R1_DATA_ROOT="${CAST_R1_DATA_ROOT:-$PROJECT_ROOT/datasets/PAPER_RL}"
export CAST_R1_SFT_MODEL="${CAST_R1_SFT_MODEL:-$PROJECT_ROOT/models/Cast-R1-SFT-8B}"
export CAST_R1_AGENT_CONFIG="$PROJECT_ROOT/recipe/time_series_forecast/base_paper.yaml"
export CAST_R1_REWARD_PATH="$PROJECT_ROOT/recipe/time_series_forecast/reward.py"
cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m recipe.time_series_forecast.paper_preflight \
  --data-root "$CAST_R1_DATA_ROOT" --sft-model "$CAST_R1_SFT_MODEL"
exec "$PYTHON_BIN" -m arft.main_agent_ppo --config-name cast_r1_paper trainer.device="$DEVICE" "$@"
