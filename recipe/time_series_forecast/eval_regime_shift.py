#!/usr/bin/env python3
"""
Regime-shift evaluation: compare Cast-R1 vs baselines on OOD subsets.

For each subset (full / high_volatility / changepoint_dense / spike),
computes MAE and MSE on the subset samples for:
  - PatchTST, iTransformer, Chronos-2 (via model_server HTTP)
  - Cast-R1 (via vLLM offline, agentic 3-turn)

Usage (baselines only, no vLLM needed):
    python eval_regime_shift.py --dataset datasets/Wind/test.parquet \
        --subsets benchmark_results/regime_shift/wind_subsets.json \
        --models patchtst,itransformer,chronos2 \
        --server-url http://localhost:8993 \
        --output benchmark_results/regime_shift/wind_baselines.json

Usage (Cast-R1, requires vLLM):
    CUDA_VISIBLE_DEVICES=0 \
    python eval_regime_shift.py --dataset datasets/Wind/test.parquet \
        --subsets benchmark_results/regime_shift/wind_subsets.json \
        --models castr1_4b \
        --llm-model models/Qwen3-4B \
        --server-url http://localhost:8993 \
        --output benchmark_results/regime_shift/wind_castr1_4b.json
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch_npu  # noqa: F401
except Exception:
    pass


from recipe.time_series_forecast.utils import parse_time_series_string


def extract_values_plain(text: str) -> list[float]:
    vals = []
    for line in text.strip().split("\n"):
        m = re.search(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+(-?\d+\.?\d*)', line)
        if m:
            vals.append(float(m.group(1)))
    return vals


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(pred: list[float], gt: list[float]) -> dict:
    min_len = min(len(pred), len(gt))
    if min_len == 0:
        return {"mae": float("nan"), "mse": float("nan"), "pred_len": len(pred), "gt_len": len(gt)}
    p, g = np.asarray(pred[:min_len]), np.asarray(gt[:min_len])
    return {
        "mae": float(np.mean(np.abs(p - g))),
        "mse": float(np.mean((p - g) ** 2)),
        "pred_len": len(pred),
        "gt_len": len(gt),
    }


def compute_normalized_metrics(pred: list[float], gt: list[float]) -> dict:
    """MAE/MSE after normalization by gt mean/std (comparable across datasets)."""
    min_len = min(len(pred), len(gt))
    if min_len == 0:
        return {"nmae": float("nan"), "nmse": float("nan")}
    p, g = np.asarray(pred[:min_len]), np.asarray(gt[:min_len])
    mu, std = np.mean(g), max(np.std(g), 1e-8)
    pn, gn = (p - mu) / std, (g - mu) / std
    return {
        "nmae": float(np.mean(np.abs(pn - gn))),
        "nmse": float(np.mean((pn - gn) ** 2)),
    }


# ---------------------------------------------------------------------------
# Baseline prediction via model_server
# ---------------------------------------------------------------------------
def predict_baseline(server_url: str, model_name: str,
                     timestamps: list[str], values: list[float],
                     horizon: int) -> list[float]:
    import requests
    resp = requests.post(f"{server_url}/predict", json={
        "timestamps": timestamps,
        "values": values,
        "model_name": model_name,
        "prediction_length": horizon,
    }, timeout=300)
    resp.raise_for_status()
    return resp.json()["values"]


# ---------------------------------------------------------------------------
# Cast-R1 prediction (vLLM offline)
# ---------------------------------------------------------------------------
_llm_engine = {}  # lazy singleton


def get_llm(model_path: str, gpu_mem: float = 0.85):
    if "llm" not in _llm_engine:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_mem,
            max_model_len=8192,
            enforce_eager=False,
        )
        _llm_engine["llm"] = llm
        _llm_engine["tokenizer"] = tokenizer
        _llm_engine["sampling_params"] = SamplingParams(temperature=0.0, max_tokens=4096)
    return _llm_engine["llm"], _llm_engine["tokenizer"], _llm_engine["sampling_params"]


def predict_castr1(model_path: str, data_str: str, server_url: str,
                   lookback: int, horizon: int, gpu_mem: float) -> list[float]:
    from recipe.time_series_forecast.benchmark_castr1 import (
        run_castr1_episode,
        extract_answer as _extract_answer,
    )
    llm, tokenizer, sp = get_llm(model_path, gpu_mem)
    result = run_castr1_episode(llm, tokenizer, sp, data_str,
                                server_url, lookback, horizon, max_steps=3)
    ans = result.get("answer", "")
    if not ans:
        return []
    return extract_values_plain(ans)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_dataset(parquet_path: str) -> pd.DataFrame:
    return pd.read_parquet(parquet_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--subsets", required=True, help="JSON from filter_regime_shift.py")
    ap.add_argument("--models", default="patchtst,itransformer,chronos2",
                    help="Comma list: patchtst,itransformer,chronos2,castr1_4b,castr1_8b")
    ap.add_argument("--server-url", default="http://localhost:8993")
    ap.add_argument("--llm-model", default=None, help="Path for castr1_* models")
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--lookback", type=int, default=96)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = load_dataset(args.dataset)
    subsets_info = json.loads(Path(args.subsets).read_text())
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    # Parse all samples upfront
    samples = {}
    for idx, row in df.iterrows():
        try:
            hist_text = row["prompt"][0]["content"]
            gt_text = row["reward_model"]["ground_truth"]
        except Exception:
            continue
        ts, vals = parse_time_series_string(hist_text)
        gt_vals = extract_values_plain(gt_text)
        if not vals or not gt_vals:
            continue
        ts_str = [t.strftime("%Y-%m-%d %H:%M:%S") if hasattr(t, "strftime") else str(t) for t in ts]
        # Truncate to lookback
        vals_trunc = vals[-args.lookback:]
        ts_trunc = ts_str[-args.lookback:]
        samples[int(idx)] = {
            "hist_text": hist_text,
            "timestamps": ts_trunc,
            "values": vals_trunc,
            "gt": gt_vals[:args.horizon],
        }
    print(f"Parsed {len(samples)} samples from {args.dataset}")

    subset_names = ["full", "high_volatility", "changepoint_dense", "spike"]
    results = {"dataset": args.dataset, "subsets": {}}

    for subset_name in subset_names:
        info = subsets_info.get(subset_name)
        if not info:
            continue
        indices = [i for i in info["indices"] if i in samples]
        print(f"\n=== Subset: {subset_name} ({len(indices)} samples) ===")
        results["subsets"][subset_name] = {"count": len(indices), "models": {}}

        for model_name in models:
            print(f"  Model: {model_name}")
            all_metrics = []
            all_nmetrics = []
            t0 = time.perf_counter()

            for i, idx in enumerate(indices):
                s = samples[idx]
                try:
                    if model_name in ("patchtst", "itransformer", "chronos2"):
                        pred = predict_baseline(
                            args.server_url, model_name,
                            s["timestamps"], s["values"], args.horizon)
                    elif model_name.startswith("castr1"):
                        if not args.llm_model:
                            print(f"    [skip] need --llm-model for {model_name}")
                            break
                        pred = predict_castr1(
                            args.llm_model, s["hist_text"],
                            args.server_url, args.lookback, args.horizon, args.gpu_mem)
                    else:
                        print(f"    [skip] unknown model {model_name}")
                        break
                    m = compute_metrics(pred, s["gt"])
                    nm = compute_normalized_metrics(pred, s["gt"])
                    all_metrics.append(m)
                    all_nmetrics.append(nm)
                except Exception as e:
                    print(f"    [ERROR] idx={idx}: {e}")
                    all_metrics.append({"mae": float("nan"), "mse": float("nan")})
                    all_nmetrics.append({"nmae": float("nan"), "nmse": float("nan")})

                if (i + 1) % 10 == 0:
                    print(f"    [{i+1}/{len(indices)}]")

            elapsed = time.perf_counter() - t0
            valid = [m for m in all_metrics if not np.isnan(m["mae"])]
            valid_n = [m for m in all_nmetrics if not np.isnan(m["nmae"])]

            if valid:
                agg = {
                    "mae": float(np.mean([m["mae"] for m in valid])),
                    "mse": float(np.mean([m["mse"] for m in valid])),
                    "nmae": float(np.mean([m["nmae"] for m in valid_n])) if valid_n else None,
                    "nmse": float(np.mean([m["nmse"] for m in valid_n])) if valid_n else None,
                    "n_valid": len(valid),
                    "n_failed": len(all_metrics) - len(valid),
                    "elapsed_s": elapsed,
                }
            else:
                agg = {"mae": None, "mse": None, "n_valid": 0, "n_failed": len(all_metrics)}
            results["subsets"][subset_name]["models"][model_name] = agg
            print(f"    → MAE={agg.get('mae','N/A'):.4f}  MSE={agg.get('mse','N/A'):.4f}" if agg.get("mae") is not None else f"    → no valid predictions")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
