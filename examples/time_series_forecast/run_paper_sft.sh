#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
device_args=()
processes="${N_GPUS_PER_NODE:-4}"
if [ "$DEVICE" = cpu ]; then
  device_args=(--cpu)
  processes="${N_GPUS_PER_NODE:-1}"
fi
exec "$PYTHON_BIN" -m torch.distributed.run --nproc_per_node="$processes" \
  -m recipe.time_series_forecast.train_policy_sft \
  --model "${MODEL_PATH:-$PROJECT_ROOT/models/Qwen3-8B}" \
  --trajectories "${TRAJECTORIES:-$PROJECT_ROOT/datasets/SFT/selected.jsonl}" \
  --output "${OUTPUT_DIR:-$PROJECT_ROOT/models/Cast-R1-SFT-8B}" "${device_args[@]}" "$@"
