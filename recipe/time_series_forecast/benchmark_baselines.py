#!/usr/bin/env python3
"""
Inference latency benchmark for the neural / foundation baselines that
Cast-R1 is compared against in the paper:

  - PatchTST       (deep learning)
  - iTransformer   (deep learning)
  - Chronos-2      (foundation model)
  - TimesFM        (foundation model, optional - only if `timesfm` is installed)

Models are loaded **in-process** via the same loaders used by `model_server.py`,
so no HTTP layer is involved and the timing reflects the pure forward pass on
the target device (CUDA by default).

Reported metrics for every model:
  * mean / p50 / p95 single-sample latency (ms)
  * single-sample throughput  (samples/s)
  * batched latency for batch sizes given by --batch-sizes
  * batched throughput        (samples/s)

Example
-------
    cd /path/to/Cast-R1
    CUDA_VISIBLE_DEVICES=0 \
        python recipe/time_series_forecast/benchmark_baselines.py \
            --dataset datasets/Wind/test.parquet \
            --device cuda \
            --num-samples 50 --num-warmup 5 \
            --batch-sizes 1,8,32 \
            --output benchmark_baselines_wind.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Make `recipe.*` importable when launched from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch_npu  # noqa: F401
except Exception:
    pass

from recipe.time_series_forecast.model_server import (  # noqa: E402
    _models,
    _resolve_device,
    _set_device_context,
    load_chronos2,
    load_itransformer,
    load_patchtst,
)
from recipe.time_series_forecast.utils import parse_time_series_string  # noqa: E402


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------
def _sync(device: str) -> None:
    if device.startswith("npu") and hasattr(torch, "npu"):
        try:
            torch.npu.synchronize()
        except Exception:
            pass
    elif device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Per-model forward functions
# ---------------------------------------------------------------------------
def _chronos2_forward(values_batch: np.ndarray, horizon: int) -> np.ndarray:
    """values_batch shape: [B, L]. Returns numpy array of forecasts."""
    pipe = _models["chronos2"]
    x = torch.tensor(values_batch, dtype=torch.float32).unsqueeze(-1)  # [B, L, 1]
    means = x.mean(1, keepdim=True)
    x = x - means
    std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
    x = x / std
    x = x.permute(0, 2, 1)  # [B, 1, L]
    quantiles, _ = pipe.predict_quantiles(
        x.cpu().numpy(),
        prediction_length=horizon,
        quantile_levels=[0.5],
    )
    return quantiles


def _torch_model_forward(name: str, values_batch: np.ndarray, _: int) -> torch.Tensor:
    model = _models[name]
    dev = next(model.parameters()).device
    x = torch.tensor(values_batch, dtype=torch.float32, device=dev).unsqueeze(-1)  # [B, L, 1]
    with torch.no_grad():
        return model(x)


def _timesfm_forward(values_batch: np.ndarray, horizon: int):
    """Optional TimesFM hook."""
    model = _models.get("timesfm")
    if model is None:
        return None
    forecasts, _ = model.forecast(
        list(values_batch),
        freq=[0] * len(values_batch),
    )
    return forecasts[..., :horizon]


def run_forward(name: str, values_batch: np.ndarray, horizon: int):
    if name == "chronos2":
        return _chronos2_forward(values_batch, horizon)
    if name == "timesfm":
        return _timesfm_forward(values_batch, horizon)
    return _torch_model_forward(name, values_batch, horizon)


# ---------------------------------------------------------------------------
# Optional TimesFM loader
# ---------------------------------------------------------------------------
def try_load_timesfm(device: str):
    try:
        import timesfm  # type: ignore
    except Exception:
        print("[skip] TimesFM not installed (`pip install timesfm`)")
        return
    try:
        backend = "gpu" if device.startswith(("cuda", "npu")) else "cpu"
        model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend=backend, per_core_batch_size=32, horizon_len=128
            ),
            checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id="google/timesfm-1.0-200m-pytorch"),
        )
        _models["timesfm"] = model
        print("TimesFM loaded.")
    except Exception as e:
        print(f"[WARN] failed to load TimesFM: {e}")


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def load_dataset(parquet_path: str, force_lookback: int | None = None) -> list[np.ndarray]:
    df = pd.read_parquet(parquet_path)
    series = []
    for _, row in df.iterrows():
        try:
            text = row["prompt"][0]["content"]
        except Exception:
            continue
        try:
            _, values = parse_time_series_string(text)
        except Exception:
            continue
        if values:
            series.append(np.asarray(values, dtype=np.float32))
    if not series:
        raise RuntimeError(f"No samples parsed from {parquet_path}")
    # Align to the most common length so we can stack into batches
    lengths = [len(v) for v in series]
    L = max(set(lengths), key=lengths.count)
    if force_lookback is not None:
        L = force_lookback
    series = [v[-L:] for v in series if len(v) >= L]
    print(f"Loaded {len(series)} samples with lookback={L} from {parquet_path}")
    return series


def benchmark_model(
    name: str,
    eval_values: list[np.ndarray],
    warm_values: list[np.ndarray],
    horizon: int,
    device: str,
    batch_sizes: list[int],
) -> dict:
    print(f"\n=== Benchmarking {name} ===")

    # warm-up (single sample)
    for v in warm_values:
        run_forward(name, v[None, :], horizon)
    _sync(device)

    # ----- single-sample latency -----
    per_sample_ms = []
    for v in eval_values:
        _sync(device)
        t0 = time.perf_counter()
        run_forward(name, v[None, :], horizon)
        _sync(device)
        per_sample_ms.append((time.perf_counter() - t0) * 1000)
    arr = np.asarray(per_sample_ms)

    result = {
        "num_samples": int(len(arr)),
        "mean_latency_ms": float(arr.mean()),
        "p50_latency_ms": float(np.percentile(arr, 50)),
        "p95_latency_ms": float(np.percentile(arr, 95)),
        "throughput_samples_per_s_bs1": float(1000.0 / arr.mean()),
        "batched": {},
    }

    # ----- batched latency -----
    for bs in batch_sizes:
        if bs <= 0 or bs > len(eval_values):
            continue
        n_iters = len(eval_values) // bs
        batches = [
            np.stack(eval_values[i * bs : (i + 1) * bs], axis=0) for i in range(n_iters)
        ]
        # warm batched
        run_forward(name, batches[0], horizon)
        _sync(device)
        t0 = time.perf_counter()
        for b in batches:
            run_forward(name, b, horizon)
        _sync(device)
        total_s = time.perf_counter() - t0
        result["batched"][str(bs)] = {
            "batch_size": int(bs),
            "num_batches": int(n_iters),
            "total_time_s": float(total_s),
            "mean_batch_latency_ms": float((total_s / n_iters) * 1000),
            "throughput_samples_per_s": float((n_iters * bs) / total_s),
        }

    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to test parquet")
    ap.add_argument("--device", default="cuda", help="cpu / cuda / npu / cuda:0 / npu:0 ...")
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--lookback", type=int, default=None,
                    help="force truncate series to this lookback (e.g. 96 for PatchTST/iTransformer)")
    ap.add_argument("--num-samples", type=int, default=50)
    ap.add_argument("--num-warmup", type=int, default=5)
    ap.add_argument("--batch-sizes", default="1,8,32")
    ap.add_argument(
        "--models",
        default="patchtst,itransformer,chronos2",
        help="comma list. Add `timesfm` if installed.",
    )
    ap.add_argument("--output", default="benchmark_baselines_results.json")
    args = ap.parse_args()

    device = _resolve_device(args.device)
    _set_device_context(device)
    print(f"Resolved device: {device}")

    models_to_run = [m.strip() for m in args.models.split(",") if m.strip()]
    loaders = {
        "chronos2": load_chronos2,
        "patchtst": load_patchtst,
        "itransformer": load_itransformer,
    }
    for name in models_to_run:
        if name == "timesfm":
            try_load_timesfm(device)
        elif name in loaders:
            loaders[name](device)
        else:
            print(f"[skip] unknown model: {name}")

    series = load_dataset(args.dataset, force_lookback=args.lookback)
    n_total = min(len(series), args.num_samples + args.num_warmup)
    warm_values = series[: args.num_warmup]
    eval_values = series[args.num_warmup : n_total]
    print(f"warm-up: {len(warm_values)}, eval: {len(eval_values)}")

    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]

    results = {
        "config": {
            "dataset": args.dataset,
            "device": device,
            "horizon": args.horizon,
            "lookback": int(len(eval_values[0])),
            "num_warmup": args.num_warmup,
            "num_samples": len(eval_values),
            "batch_sizes": batch_sizes,
        },
        "models": {},
    }

    for name in models_to_run:
        if _models.get(name) is None:
            print(f"[skip] {name}: model not loaded")
            continue
        results["models"][name] = benchmark_model(
            name=name,
            eval_values=eval_values,
            warm_values=warm_values,
            horizon=args.horizon,
            device=device,
            batch_sizes=batch_sizes,
        )

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
