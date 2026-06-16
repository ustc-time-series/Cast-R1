#!/usr/bin/env python3
"""Evaluate ETTh2 baselines on AgentRFT parquet forecast windows.

This script keeps all evaluated models on the same input windows and reports
metrics in the original target scale. It is intended for quick Cast-R1-style
baseline checks where the dataset is already represented as RL parquet rows.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recipe.time_series_forecast.utils import parse_time_series_string  # noqa: E402


def extract_values_plain(text: str) -> list[float]:
    values: list[float] = []
    for line in str(text).strip().splitlines():
        match = re.search(
            r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+(-?\d+(?:\.\d+)?)",
            line,
        )
        if match:
            values.append(float(match.group(1)))
    return values


def compute_metrics(pred: list[float], gt: list[float]) -> dict[str, Any]:
    n = min(len(pred), len(gt))
    if n == 0:
        return {"mse": math.nan, "mae": math.nan, "pred_len": len(pred), "gt_len": len(gt)}
    p = np.asarray(pred[:n], dtype=np.float64)
    g = np.asarray(gt[:n], dtype=np.float64)
    return {
        "mse": float(np.mean((p - g) ** 2)),
        "mae": float(np.mean(np.abs(p - g))),
        "pred_len": len(pred),
        "gt_len": len(gt),
    }


def load_samples(parquet_path: Path, lookback: int, horizon: int, limit: int | None) -> list[dict[str, Any]]:
    df = pd.read_parquet(parquet_path)
    samples: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        try:
            hist_text = row["prompt"][0]["content"]
            gt_text = row["reward_model"]["ground_truth"]
        except Exception:
            continue

        timestamps, values = parse_time_series_string(hist_text)
        gt_values = extract_values_plain(gt_text)
        if not timestamps or not values or not gt_values:
            continue

        ts_str = [
            ts.strftime("%Y-%m-%d %H:%M:%S") if hasattr(ts, "strftime") else str(ts)
            for ts in timestamps
        ]
        samples.append(
            {
                "idx": int(idx),
                "timestamps": ts_str[-lookback:],
                "values": [float(v) for v in values[-lookback:]],
                "gt": [float(v) for v in gt_values[:horizon]],
            }
        )
        if limit is not None and len(samples) >= limit:
            break
    if not samples:
        raise RuntimeError(f"No samples parsed from {parquet_path}")
    return samples


def predict_model_server(
    server_url: str,
    model_name: str,
    sample: dict[str, Any],
    horizon: int,
    dataset_name: str,
) -> list[float]:
    response = requests.post(
        f"{server_url}/predict",
        json={
            "timestamps": sample["timestamps"],
            "values": sample["values"],
            "model_name": model_name,
            "prediction_length": horizon,
            "dataset_name": dataset_name,
            "data_source": dataset_name,
        },
        timeout=300,
    )
    response.raise_for_status()
    return [float(v) for v in response.json()["values"][:horizon]]


def predict_arima(sample: dict[str, Any], horizon: int) -> list[float]:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller

    values = np.asarray(sample["values"], dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            adf_result = adfuller(values, maxlag=min(10, len(values) // 3))
            d = 0 if adf_result[1] < 0.05 else 1
        except Exception:
            d = 1

        configs = [
            (1, d, 1),
            (2, d, 1),
            (1, d, 2),
            (2, d, 2),
            (3, d, 1),
            (1, d, 3),
            (0, d, 1),
            (1, d, 0),
            (5, d, 0),
            (0, d, 5),
        ]
        best_model = None
        best_aic = float("inf")
        for order in configs:
            try:
                fitted = ARIMA(values, order=order).fit()
                if fitted.aic < best_aic:
                    best_aic = float(fitted.aic)
                    best_model = fitted
            except Exception:
                continue
        if best_model is None:
            return [float(values[-1])] * horizon
        forecast = best_model.forecast(steps=horizon)
    return [float(v) for v in np.asarray(forecast)[:horizon]]


def predict_prophet(sample: dict[str, Any], horizon: int) -> list[float]:
    from prophet import Prophet

    train_df = pd.DataFrame(
        {
            "ds": pd.to_datetime(sample["timestamps"]),
            "y": np.asarray(sample["values"], dtype=np.float64),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
        )
        model.fit(train_df)
        future = model.make_future_dataframe(periods=horizon, freq="h", include_history=False)
        forecast = model.predict(future)
    return [float(v) for v in forecast["yhat"].to_numpy()[:horizon]]


_TIMESFM_MODEL = None


def load_timesfm_model():
    global _TIMESFM_MODEL
    if _TIMESFM_MODEL is not None:
        return _TIMESFM_MODEL

    import timesfm

    _TIMESFM_MODEL = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="cpu",
            per_core_batch_size=1,
            horizon_len=128,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
        ),
    )
    return _TIMESFM_MODEL


def predict_timesfm(sample: dict[str, Any], horizon: int) -> list[float]:
    model = load_timesfm_model()
    values = np.asarray(sample["values"], dtype=np.float32)
    forecasts, _ = model.forecast([values], freq=[0])
    return [float(v) for v in np.asarray(forecasts)[0, :horizon]]


def predict(model_name: str, sample: dict[str, Any], args: argparse.Namespace) -> list[float]:
    if model_name in {"chronos2", "patchtst", "itransformer"}:
        return predict_model_server(
            args.server_url,
            model_name,
            sample,
            args.horizon,
            args.dataset_name,
        )
    if model_name == "arima":
        return predict_arima(sample, args.horizon)
    if model_name == "prophet":
        return predict_prophet(sample, args.horizon)
    if model_name == "timesfm":
        return predict_timesfm(sample, args.horizon)
    raise ValueError(f"Unsupported model: {model_name}")


def aggregate(per_sample: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [item for item in per_sample if not math.isnan(item["metrics"]["mse"])]
    if not valid:
        return {"mse": math.nan, "mae": math.nan, "n_valid": 0, "n_failed": len(per_sample)}
    return {
        "mse": float(np.mean([item["metrics"]["mse"] for item in valid])),
        "mae": float(np.mean([item["metrics"]["mae"] for item in valid])),
        "n_valid": len(valid),
        "n_failed": len(per_sample) - len(valid),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "RL" / "ETTH2" / "test.parquet",
    )
    parser.add_argument("--dataset-name", default="ETTH2")
    parser.add_argument("--server-url", default="http://localhost:8993")
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--models",
        default="arima,prophet,chronos2,patchtst,itransformer,timesfm",
        help="Comma-separated: arima,prophet,chronos2,patchtst,itransformer,timesfm",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "benchmark_results" / "etth2_cast_r1" / "window_baselines.json",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.dataset, args.lookback, args.horizon, args.limit)
    model_names = [name.strip().lower() for name in args.models.split(",") if name.strip()]

    result: dict[str, Any] = {
        "config": {
            "dataset": str(args.dataset),
            "dataset_name": args.dataset_name,
            "lookback": args.lookback,
            "horizon": args.horizon,
            "num_samples": len(samples),
            "server_url": args.server_url,
        },
        "models": {},
    }

    for model_name in model_names:
        print(f"\n=== {model_name} ===", flush=True)
        per_sample: list[dict[str, Any]] = []
        start = time.perf_counter()
        for pos, sample in enumerate(samples, start=1):
            try:
                pred = predict(model_name, sample, args)
                metrics = compute_metrics(pred, sample["gt"])
                error = None
            except Exception as exc:
                pred = []
                metrics = {"mse": math.nan, "mae": math.nan, "pred_len": 0, "gt_len": len(sample["gt"])}
                error = repr(exc)
                print(f"[{model_name}] sample {sample['idx']} failed: {error}", flush=True)
            per_sample.append(
                {
                    "idx": sample["idx"],
                    "metrics": metrics,
                    "error": error,
                }
            )
            if pos % 5 == 0 or pos == len(samples):
                print(f"[{model_name}] {pos}/{len(samples)}", flush=True)

        summary = aggregate(per_sample)
        summary["elapsed_s"] = float(time.perf_counter() - start)
        result["models"][model_name] = {
            "summary": summary,
            "per_sample": per_sample,
        }
        args.output.write_text(json.dumps(result, indent=2, allow_nan=True))
        print(f"{model_name}: MSE={summary['mse']:.6f} MAE={summary['mae']:.6f}", flush=True)

    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
