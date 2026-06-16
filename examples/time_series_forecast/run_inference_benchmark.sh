#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATASET_NAME="${1:-wind}"
case "$DATASET_NAME" in
    wind)  DATA_PATH="${DATA_PATH:-$PROJECT_ROOT/datasets/Wind/test.parquet}" ;;
    etth1) DATA_PATH="${DATA_PATH:-$PROJECT_ROOT/datasets/ETTH1/test.parquet}" ;;
    ettm1) DATA_PATH="${DATA_PATH:-$PROJECT_ROOT/datasets/ETTM1/test.parquet}" ;;
    np)    DATA_PATH="${DATA_PATH:-$PROJECT_ROOT/datasets/NP/test.parquet}" ;;
    pjm)   DATA_PATH="${DATA_PATH:-$PROJECT_ROOT/datasets/PJM/test.parquet}" ;;
    *)     echo "Unknown dataset: $DATASET_NAME" >&2; exit 1 ;;
esac

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
SERVER_DEVICE="${SERVER_DEVICE:-$DEVICE}"
SERVER_VISIBLE_DEVICES="${SERVER_VISIBLE_DEVICES:-0}"
BASELINE_VISIBLE_DEVICES="${BASELINE_VISIBLE_DEVICES:-1}"
CASTR1_4B_VISIBLE_DEVICES="${CASTR1_4B_VISIBLE_DEVICES:-2}"
CASTR1_8B_VISIBLE_DEVICES="${CASTR1_8B_VISIBLE_DEVICES:-3}"
SERVER_PORT="${SERVER_PORT:-8993}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:${SERVER_PORT}}"
RESULT_DIR="${RESULT_DIR:-$PROJECT_ROOT/benchmark_results/$DATASET_NAME}"
MODEL_4B_PATH="${MODEL_4B_PATH:-$PROJECT_ROOT/models/Qwen3-4B}"
MODEL_8B_PATH="${MODEL_8B_PATH:-$PROJECT_ROOT/models/Qwen3-8B}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
NUM_WARMUP="${NUM_WARMUP:-5}"
LOOKBACK="${LOOKBACK:-96}"
HORIZON="${HORIZON:-96}"
TIME_REASONER_TP="${TIME_REASONER_TP:-1}"
CASTR1_4B_TP="${CASTR1_4B_TP:-1}"
CASTR1_8B_TP="${CASTR1_8B_TP:-1}"

mkdir -p "$RESULT_DIR"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if ! curl -fsS "$SERVER_URL/health" >/dev/null 2>&1; then
    echo "[INFO] model server not detected on $SERVER_URL; starting a local server"
    nohup env \
        DEVICE="$SERVER_DEVICE" \
        VISIBLE_DEVICES="$SERVER_VISIBLE_DEVICES" \
        PORT="$SERVER_PORT" \
        PYTHON_BIN="$PYTHON_BIN" \
        bash "$PROJECT_ROOT/recipe/time_series_forecast/start_model_server.sh" \
        > "$RESULT_DIR/model_server.log" 2>&1 &
    for _ in $(seq 1 120); do
        sleep 1
        if curl -fsS "$SERVER_URL/health" >/dev/null 2>&1; then
            break
        fi
    done
fi

curl -fsS "$SERVER_URL/health" >/dev/null

echo "===== [1/3] Baselines ====="
env CUDA_VISIBLE_DEVICES="$BASELINE_VISIBLE_DEVICES" \
    "$PYTHON_BIN" "$PROJECT_ROOT/recipe/time_series_forecast/benchmark_baselines.py" \
    --dataset "$DATA_PATH" \
    --device "$DEVICE" \
    --horizon "$HORIZON" \
    --num-samples "$NUM_SAMPLES" \
    --num-warmup "$NUM_WARMUP" \
    --batch-sizes 1,8,32 \
    --models patchtst,itransformer,chronos2 \
    --output "$RESULT_DIR/baselines.json"

echo "===== [2/3] TimeReasoner-style ====="
env CUDA_VISIBLE_DEVICES="$CASTR1_8B_VISIBLE_DEVICES" \
    "$PYTHON_BIN" "$PROJECT_ROOT/recipe/time_series_forecast/benchmark_castr1.py" \
    --mode time-reasoner \
    --model "$MODEL_8B_PATH" \
    --dataset "$DATA_PATH" \
    --tp "$TIME_REASONER_TP" \
    --num-samples "$NUM_SAMPLES" \
    --num-warmup "$NUM_WARMUP" \
    --lookback "$LOOKBACK" \
    --horizon "$HORIZON" \
    --output "$RESULT_DIR/time_reasoner_qwen8b.json"

echo "===== [3/3] Cast-R1 ====="
env CUDA_VISIBLE_DEVICES="$CASTR1_4B_VISIBLE_DEVICES" \
    "$PYTHON_BIN" "$PROJECT_ROOT/recipe/time_series_forecast/benchmark_castr1.py" \
    --mode cast-r1 \
    --model "$MODEL_4B_PATH" \
    --dataset "$DATA_PATH" \
    --tp "$CASTR1_4B_TP" \
    --server-url "$SERVER_URL" \
    --num-samples "$NUM_SAMPLES" \
    --num-warmup "$NUM_WARMUP" \
    --lookback "$LOOKBACK" \
    --horizon "$HORIZON" \
    --max-steps 3 \
    --output "$RESULT_DIR/castr1_qwen4b.json"

env CUDA_VISIBLE_DEVICES="$CASTR1_8B_VISIBLE_DEVICES" \
    "$PYTHON_BIN" "$PROJECT_ROOT/recipe/time_series_forecast/benchmark_castr1.py" \
    --mode cast-r1 \
    --model "$MODEL_8B_PATH" \
    --dataset "$DATA_PATH" \
    --tp "$CASTR1_8B_TP" \
    --server-url "$SERVER_URL" \
    --num-samples "$NUM_SAMPLES" \
    --num-warmup "$NUM_WARMUP" \
    --lookback "$LOOKBACK" \
    --horizon "$HORIZON" \
    --max-steps 3 \
    --output "$RESULT_DIR/castr1_qwen8b.json"

echo "Results written to $RESULT_DIR"
