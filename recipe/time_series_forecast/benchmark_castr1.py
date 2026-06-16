#!/usr/bin/env python3
"""
Inference latency benchmark for Cast-R1 and an LLM-only TimeReasoner-style
baseline, using vLLM offline inference.

Modes
-----
  --mode cast-r1
      Replicates the 3-turn agentic workflow used in
      `time_series_forecast_agent_flow.py` (feature extraction → predict →
      reflect/answer). Tool calls go through the existing model server (HTTP)
      so the timing reflects the actual production path.

  --mode time-reasoner
      A single-turn LLM-only forecast (no tools, no model server). Used as
      a stand-in for TimeReasoner-style baselines, which only consume the
      LLM's reasoning ability.

Reported metrics
----------------
  * mean / p50 / p95 single-sample latency (s)
  * average tokens generated per sample
  * average #LLM turns and #tool calls per sample (cast-r1 only)
  * end-to-end throughput (samples / s)

Example
-------
    cd /path/to/Cast-R1
    # 1. start the model server on a separate device (for cast-r1 tool calls)
    bash recipe/time_series_forecast/start_model_server.sh 0 8993 cuda

    # 2. run the agent benchmark on Qwen3-8B (TP=2)
    CUDA_VISIBLE_DEVICES=1,2 \
        python recipe/time_series_forecast/benchmark_castr1.py \
            --mode cast-r1 \
            --model models/Qwen3-8B \
            --dataset datasets/Wind/test.parquet \
            --tp 2 --num-samples 20 --num-warmup 2 \
            --server-url http://localhost:8993 \
            --output benchmark_castr1_qwen8b_wind.json
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch_npu  # noqa: F401
except Exception:
    pass

from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

from recipe.time_series_forecast.prompts import (  # noqa: E402
    TIMESERIES_SYSTEM_PROMPT,
    TIMESERIES_TOOL_SCHEMAS,
)
from recipe.time_series_forecast.utils import (  # noqa: E402
    extract_basic_statistics,
    extract_data_quality,
    extract_event_summary,
    extract_forecast_residuals,
    extract_within_channel_dynamics,
    format_basic_statistics,
    format_data_quality,
    format_event_summary,
    format_forecast_residuals,
    format_within_channel_dynamics,
    get_last_timestamp,
    parse_time_series_string,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("benchmark_castr1")


# ---------------------------------------------------------------------------
# Tool execution (mirrors recipe/time_series_forecast/time_series_forecast_agent_flow.py)
# ---------------------------------------------------------------------------
EXTRACT_TOOL_FNS = {
    "extract_basic_statistics": (extract_basic_statistics, format_basic_statistics),
    "extract_within_channel_dynamics": (extract_within_channel_dynamics, format_within_channel_dynamics),
    "extract_forecast_residuals": (extract_forecast_residuals, format_forecast_residuals),
    "extract_data_quality": (extract_data_quality, format_data_quality),
    "extract_event_summary": (extract_event_summary, format_event_summary),
}


def _format_predictions_to_string(timestamps: list[str], values: list[float]) -> str:
    lines = []
    for ts, v in zip(timestamps, values):
        try:
            lines.append(f"{ts} {float(v):.3f}")
        except (TypeError, ValueError):
            lines.append(f"{ts} {v}")
    return "\n".join(lines)


def _call_predict_server(
    server_url: str,
    timestamps: list[str],
    values: list[float],
    horizon: int,
    model_name: str,
    timeout: int = 300,
) -> str:
    if model_name not in {"chronos2", "patchtst", "itransformer", "arima"}:
        model_name = "chronos2"
    resp = requests.post(
        f"{server_url}/predict",
        json={
            "timestamps": timestamps,
            "values": values,
            "model_name": model_name,
            "prediction_length": horizon,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    out = resp.json()
    return _format_predictions_to_string(out["timestamps"], out["values"])


def execute_tool(
    name: str,
    arguments: dict,
    ctx: dict,
) -> str:
    if name in EXTRACT_TOOL_FNS:
        extract_fn, format_fn = EXTRACT_TOOL_FNS[name]
        feats = extract_fn(data=ctx["values"])
        ctx["features_done"] = True
        return format_fn(feats)
    if name == "predict_time_series":
        if not ctx.get("features_done"):
            return "ERROR: please call a feature extraction tool before predict_time_series"
        model_name = (arguments or {}).get("model_name", "chronos2")
        return _call_predict_server(
            server_url=ctx["server_url"],
            timestamps=ctx["timestamps_str"],
            values=ctx["values"],
            horizon=ctx["horizon"],
            model_name=model_name,
        )
    return f"ERROR: unknown tool {name}"


# ---------------------------------------------------------------------------
# Hermes-style tool-call parsing
# ---------------------------------------------------------------------------
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def parse_tool_calls(text: str) -> list[tuple[str, dict]]:
    calls = []
    for m in TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
            args = obj.get("arguments", {}) or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            calls.append((obj.get("name", ""), args))
        except Exception:
            continue
    return calls


def extract_answer(text: str) -> str | None:
    m = ANSWER_RE.search(text)
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# Prompt builders (mirror agent_flow)
# ---------------------------------------------------------------------------
def _format_history(history: list[str]) -> str:
    return "\n".join(history) if history else "No previous analysis performed."


def _truncate_data(data_str: str, head: int = 5, tail: int = 5) -> str:
    lines = data_str.strip().split("\n")
    if len(lines) <= head + tail:
        return data_str
    omitted = len(lines) - head - tail
    return "\n".join(lines[:head]) + f"\n... ({omitted} rows omitted) ...\n" + "\n".join(lines[-tail:])


def build_user_prompt(
    data_str: str,
    lookback: int,
    horizon: int,
    history: list[str],
    prediction: str | None,
) -> str:
    history_text = _format_history(history)
    if prediction:
        truncated = _truncate_data(data_str)
        return (
            f"**[Turn 3] Action: Output your final answer in <think>...</think><answer>...</answer> format using the model predictions. Do NOT output any text outside these tags.**\n"
            f"### Lookback Window: {lookback} rows\n"
            f"### Forecast Horizon: {horizon} rows\n"
            f"### Historical Data (truncated)\n{truncated}\n\n"
            f"### Analysis History\n{history_text}\n\n"
            f"### Model Predictions\nModel Predictions ({horizon} steps):\n{prediction}\n\n"
            "**Instructions**:\n"
            "1. Reflect on whether the Model Predictions are reasonable based on the historical patterns\n"
            "2. If predictions look reasonable -> copy the predictions into <answer>\n"
            "3. If predictions need adjustment -> make minor modifications based on your analysis\n"
            "4. Your final answer should stay close to the Model Predictions\n"
            "5. Output ONLY <think>...</think> and <answer>...</answer>, nothing else\n"
        )

    if not history:
        action = "Call feature extraction tools (e.g., extract_basic_statistics). Do NOT call predict_time_series yet."
        turn = 1
    else:
        action = "Call predict_time_series with your chosen model (e.g., 'chronos2')."
        turn = 2
    return (
        f"**[Turn {turn}] Action: {action}**\n"
        f"### Lookback Window: {lookback} rows\n"
        f"### Forecast Horizon: {horizon} rows\n"
        f"### Historical Data\n{data_str}\n\n"
        f"### Analysis History\n{history_text}\n\n"
        "### Model Predictions\nNo predictions available yet. Call predict_time_series to generate forecasts.\n"
    )


TIME_REASONER_SYSTEM = (
    "You are a time series forecasting assistant. Reason carefully about the "
    "trend, seasonality and recent dynamics of the input series, then output "
    "your final forecast inside <answer>...</answer>, one '<timestamp> <value>' "
    "pair per line. Do not output any text outside <think>...</think> and "
    "<answer>...</answer>."
)


def build_time_reasoner_prompt(data_str: str, lookback: int, horizon: int) -> str:
    return (
        f"### Lookback Window: {lookback} rows\n"
        f"### Forecast Horizon: {horizon} rows\n"
        f"### Historical Data\n{data_str}\n\n"
        "Predict the next "
        f"{horizon} values. Output exactly {horizon} timestamp value pairs inside "
        "<answer>...</answer>, continuing the historical timestamp grid."
    )


# ---------------------------------------------------------------------------
# Episode runners
# ---------------------------------------------------------------------------
def run_castr1_episode(
    llm: LLM,
    tokenizer: AutoTokenizer,
    sampling_params: SamplingParams,
    data_str: str,
    server_url: str,
    lookback: int,
    horizon: int,
    max_steps: int = 3,
) -> dict:
    timestamps, values = parse_time_series_string(data_str)
    timestamps_str = [
        ts.strftime("%Y-%m-%d %H:%M:%S") if hasattr(ts, "strftime") else str(ts)
        for ts in timestamps
    ]
    ctx = {
        "values": values,
        "timestamps_str": timestamps_str,
        "horizon": horizon,
        "server_url": server_url,
        "features_done": False,
    }
    history: list[str] = []
    prediction_str: str | None = None

    n_llm_turns = 0
    n_tool_calls = 0
    n_gen_tokens = 0
    answer = None

    for _ in range(max_steps):
        user_prompt = build_user_prompt(data_str, lookback, horizon, history, prediction_str)
        messages = [
            {"role": "system", "content": TIMESERIES_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tools=TIMESERIES_TOOL_SCHEMAS,
            add_generation_prompt=True,
            tokenize=False,
        )
        outputs = llm.generate([prompt_text], sampling_params, use_tqdm=False)
        out0 = outputs[0].outputs[0]
        text = out0.text
        n_llm_turns += 1
        n_gen_tokens += len(out0.token_ids)

        ans = extract_answer(text)
        if ans:
            answer = ans
            break

        for name, args in parse_tool_calls(text):
            try:
                tool_result = execute_tool(name, args, ctx)
            except Exception as e:
                tool_result = f"ERROR: {e}"
            n_tool_calls += 1
            if name == "predict_time_series" and not tool_result.startswith("ERROR"):
                prediction_str = tool_result
                history.append(f"Model Prediction generated using {args.get('model_name', 'chronos2')}")
            elif name in EXTRACT_TOOL_FNS:
                history.append(tool_result)
        # if no progress was made, abort early
        if not history and prediction_str is None:
            break

    return {
        "answer": answer,
        "n_llm_turns": n_llm_turns,
        "n_tool_calls": n_tool_calls,
        "n_gen_tokens": n_gen_tokens,
    }


def run_time_reasoner_episode(
    llm: LLM,
    tokenizer: AutoTokenizer,
    sampling_params: SamplingParams,
    data_str: str,
    lookback: int,
    horizon: int,
) -> dict:
    messages = [
        {"role": "system", "content": TIME_REASONER_SYSTEM},
        {"role": "user", "content": build_time_reasoner_prompt(data_str, lookback, horizon)},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    outputs = llm.generate([prompt_text], sampling_params, use_tqdm=False)
    out0 = outputs[0].outputs[0]
    return {
        "answer": extract_answer(out0.text),
        "n_llm_turns": 1,
        "n_tool_calls": 0,
        "n_gen_tokens": len(out0.token_ids),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def load_samples(parquet_path: str, total_needed: int) -> list[str]:
    df = pd.read_parquet(parquet_path)
    out: list[str] = []
    for _, row in df.iterrows():
        try:
            out.append(row["prompt"][0]["content"])
        except Exception:
            continue
        if len(out) >= total_needed:
            break
    if not out:
        raise RuntimeError(f"No samples loaded from {parquet_path}")
    print(f"Loaded {len(out)} samples from {parquet_path}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cast-r1", "time-reasoner"], default="cast-r1")
    ap.add_argument("--model", required=True, help="Path or HF id of the LLM (e.g. Qwen3-8B)")
    ap.add_argument("--dataset", required=True, help="Path to test parquet")
    ap.add_argument("--server-url", default="http://localhost:8993")
    ap.add_argument("--tp", type=int, default=1, help="vLLM tensor_parallel_size")
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--num-warmup", type=int, default=2)
    ap.add_argument("--lookback", type=int, default=96)
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--max-steps", type=int, default=3)
    ap.add_argument("--max-model-len", type=int, default=12288)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--output", default="benchmark_castr1_results.json")
    args = ap.parse_args()

    if args.mode == "cast-r1":
        try:
            h = requests.get(f"{args.server_url}/health", timeout=5).json()
            print(f"model server health: {h}")
        except Exception as e:
            print(f"[FATAL] cast-r1 mode requires the model server. Failed to reach {args.server_url}: {e}")
            sys.exit(1)

    samples = load_samples(args.dataset, args.num_samples + args.num_warmup)
    warm = samples[: args.num_warmup]
    eval_samples = samples[args.num_warmup : args.num_warmup + args.num_samples]

    print(f"Loading vLLM with model={args.model} tp={args.tp}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        enforce_eager=False,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    def _run_one(data_str: str) -> dict:
        if args.mode == "cast-r1":
            return run_castr1_episode(
                llm, tokenizer, sampling_params, data_str,
                args.server_url, args.lookback, args.horizon,
                max_steps=args.max_steps,
            )
        return run_time_reasoner_episode(
            llm, tokenizer, sampling_params, data_str,
            args.lookback, args.horizon,
        )

    # warm up
    print(f"Warming up with {len(warm)} samples ...")
    for s in warm:
        _run_one(s)

    print(f"Benchmarking on {len(eval_samples)} samples ...")
    latencies: list[float] = []
    metas: list[dict] = []
    t_start = time.perf_counter()
    for i, s in enumerate(eval_samples):
        t0 = time.perf_counter()
        meta = _run_one(s)
        dt = time.perf_counter() - t0
        latencies.append(dt)
        metas.append(meta)
        print(
            f"  [{i+1:3d}/{len(eval_samples)}] {dt:6.2f}s "
            f"turns={meta['n_llm_turns']} tools={meta['n_tool_calls']} "
            f"gen_tok={meta['n_gen_tokens']} ans={'Y' if meta['answer'] else 'N'}"
        )
    total = time.perf_counter() - t_start

    arr = np.asarray(latencies)
    result = {
        "config": {
            "mode": args.mode,
            "model": args.model,
            "dataset": args.dataset,
            "tp": args.tp,
            "num_warmup": args.num_warmup,
            "num_samples": len(arr),
            "lookback": args.lookback,
            "horizon": args.horizon,
            "max_steps": args.max_steps,
            "temperature": args.temperature,
        },
        "mean_latency_s": float(arr.mean()),
        "p50_latency_s": float(np.percentile(arr, 50)),
        "p95_latency_s": float(np.percentile(arr, 95)),
        "throughput_samples_per_s": float(len(arr) / total),
        "total_time_s": float(total),
        "mean_llm_turns": float(np.mean([m["n_llm_turns"] for m in metas])),
        "mean_tool_calls": float(np.mean([m["n_tool_calls"] for m in metas])),
        "mean_gen_tokens": float(np.mean([m["n_gen_tokens"] for m in metas])),
        "answer_success_rate": float(np.mean([1 if m["answer"] else 0 for m in metas])),
    }
    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2))
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
