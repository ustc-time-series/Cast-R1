#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

VISIBLE_DEVICES="${1:-${VISIBLE_DEVICES:-0}}"
PORT="${2:-${PORT:-8993}}"
DEVICE="${3:-${DEVICE:-cuda}}"
HOST="${HOST:-0.0.0.0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$SCRIPT_DIR"
if [[ "$DEVICE" == cuda* ]]; then
  exec env CUDA_VISIBLE_DEVICES="$VISIBLE_DEVICES" \
    "$PYTHON_BIN" model_server.py --host "$HOST" --port "$PORT" --device "$DEVICE"
fi

exec "$PYTHON_BIN" model_server.py --host "$HOST" --port "$PORT" --device "$DEVICE"
