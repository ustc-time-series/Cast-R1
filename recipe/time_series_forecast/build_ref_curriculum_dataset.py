#!/usr/bin/env python3
"""Build reference-band curriculum parquet files for time-series RL.

The output keeps prompts, labels, and sample order unchanged. Only
``extra_info.band`` is replaced with train-only reference bands.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from recipe.time_series_forecast.reward import find_change_points
except Exception:  # pragma: no cover - keeps the CLI usable in minimal envs.
    find_change_points = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_SPLITS = ("train", "val", "test")


def extract_values(text: str) -> list[float]:
    values: list[float] = []
    for line in str(text).strip().splitlines():
        matches = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", line)
        if matches:
            values.append(float(matches[-1]))
    return values


def _robust_scale(values: np.ndarray) -> float:
    if values.size == 0:
        return 1.0

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    std = float(np.std(values))
    scale = max(mad * 1.4826, std, abs(median) * 0.05, 1e-6)
    return scale


def compute_reference_value(history: Iterable[float], future: Iterable[float]) -> float:
    """Return a train-only hardness/reference score.

    The score combines last-value naive forecast error with future movement.
    It intentionally uses only the sample's training label and context, never
    validation/test distribution statistics.
    """

    hist = np.asarray(list(history), dtype=float)
    fut = np.asarray(list(future), dtype=float)
    if hist.size == 0 or fut.size == 0:
        return 0.0

    last = float(hist[-1])
    naive = np.full_like(fut, last)
    mae = float(np.mean(np.abs(fut - naive)))
    mse = float(np.mean((fut - naive) ** 2))
    future_diff = np.diff(fut)
    spike = float(np.max(np.abs(future_diff))) if future_diff.size else 0.0
    future_mad = float(np.mean(np.abs(fut - np.median(fut))))
    hist_std = float(np.std(hist)) + 1e-6
    rel_vol = future_mad / hist_std

    if find_change_points is not None and fut.size >= 5:
        peaks, valleys = find_change_points(fut.tolist())
        changepoints = float(len(peaks) + len(valleys)) / float(fut.size)
    else:
        signs = np.sign(future_diff)
        changepoints = float(np.sum(signs[1:] * signs[:-1] < 0)) / float(max(fut.size, 1))

    components = (
        math.log1p(mae),
        0.5 * math.log1p(mse),
        0.5 * math.log1p(spike),
        0.25 * math.log1p(rel_vol),
        0.25 * changepoints,
    )
    return float(sum(components))


def compute_normalized_reference_value(history: Iterable[float], future: Iterable[float]) -> float:
    """Return a scale-robust hardness score aligned with normalized rewards."""

    hist = np.asarray(list(history), dtype=float)
    fut = np.asarray(list(future), dtype=float)
    if hist.size == 0 or fut.size == 0:
        return 0.0

    last = float(hist[-1])
    naive = np.full_like(fut, last)
    scale = _robust_scale(np.concatenate([hist, fut]))

    abs_err = np.abs(fut - naive) / scale
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(abs_err**2)))

    future_diff = np.diff(fut)
    spike = float(np.max(np.abs(future_diff)) / scale) if future_diff.size else 0.0
    future_mad = float(np.mean(np.abs(fut - np.median(fut))) / scale)

    hist_diff = np.diff(hist)
    hist_step_scale = _robust_scale(hist_diff) if hist_diff.size else scale
    relative_spike = float(np.max(np.abs(future_diff)) / hist_step_scale) if future_diff.size else 0.0

    if find_change_points is not None and fut.size >= 5:
        peaks, valleys = find_change_points(fut.tolist())
        changepoints = float(len(peaks) + len(valleys)) / float(fut.size)
    else:
        signs = np.sign(future_diff)
        changepoints = float(np.sum(signs[1:] * signs[:-1] < 0)) / float(max(fut.size, 1))

    components = (
        math.log1p(mae),
        0.5 * math.log1p(rmse),
        0.35 * math.log1p(spike),
        0.25 * math.log1p(future_mad),
        0.20 * math.log1p(relative_spike),
        0.25 * changepoints,
    )
    return float(sum(components))


def assign_quantile_bands(values: Iterable[float]) -> list[int]:
    indexed = list(enumerate(float(value) for value in values))
    if not indexed:
        return []

    ordered = sorted(indexed, key=lambda item: (item[1], item[0]))
    n = len(ordered)
    bands = [1] * n
    for rank, (original_index, _) in enumerate(ordered):
        if rank < n / 3:
            band = 1
        elif rank < 2 * n / 3:
            band = 2
        else:
            band = 3
        bands[original_index] = band
    return bands


def _history_text(row: pd.Series) -> str:
    prompt = row["prompt"]
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()
    if isinstance(prompt, list) and prompt:
        first = prompt[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
    raise ValueError("row prompt does not contain prompt[0]['content']")


def _ground_truth_text(row: pd.Series) -> str:
    reward_model = row["reward_model"]
    if isinstance(reward_model, dict):
        return str(reward_model.get("ground_truth", ""))
    raise ValueError("row reward_model does not contain ground_truth")


def annotate_dataframe(df: pd.DataFrame, split: str, *, method: str = "naive_error_variation") -> tuple[pd.DataFrame, dict]:
    values: list[float] = []
    for _, row in df.iterrows():
        history = extract_values(_history_text(row))
        future = extract_values(_ground_truth_text(row))
        if method == "normalized_naive_error_variation":
            values.append(compute_normalized_reference_value(history, future))
        elif method == "naive_error_variation":
            values.append(compute_reference_value(history, future))
        else:
            raise ValueError(f"Unknown reference band method: {method}")

    if split == "train":
        bands = assign_quantile_bands(values)
    else:
        bands = [
            int((dict(extra).get("band") if isinstance(extra, dict) and "band" in extra else 2))
            for extra in df["extra_info"]
        ]

    out = df.copy()
    new_extra = []
    for extra, band, reference_value in zip(out["extra_info"], bands, values, strict=True):
        info = dict(extra) if isinstance(extra, dict) else {}
        info["band"] = int(band)
        info["ref_band_reference_value"] = float(reference_value)
        info["ref_band_method"] = f"{method}_train_quantile"
        new_extra.append(info)
    out["extra_info"] = new_extra

    counts = {str(band): int(sum(1 for item in bands if item == band)) for band in (1, 2, 3)}
    stats = {
        "count": int(len(values)),
        "band_counts": counts,
        "reference_min": float(np.min(values)) if values else None,
        "reference_max": float(np.max(values)) if values else None,
        "reference_mean": float(np.mean(values)) if values else None,
    }
    return out, stats


def build_dataset(input_root: Path, output_root: Path, dataset: str, *, method: str = "naive_error_variation") -> dict:
    dataset_manifest: dict = {"dataset": dataset, "splits": {}}
    src_dir = input_root / dataset
    dst_dir = output_root / dataset
    dst_dir.mkdir(parents=True, exist_ok=True)

    for split in SUPPORTED_SPLITS:
        src = src_dir / f"{split}.parquet"
        if not src.exists():
            continue
        df = pd.read_parquet(src)
        out, stats = annotate_dataframe(df, split=split, method=method)
        out.to_parquet(dst_dir / f"{split}.parquet", index=False)
        dataset_manifest["splits"][split] = stats

    return dataset_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "RL_CURRICULUM",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "RL_CURRICULUM_REF_NAIVEVAR",
    )
    parser.add_argument("--datasets", nargs="+", default=["NP", "WIND"])
    parser.add_argument(
        "--method",
        choices=["naive_error_variation", "normalized_naive_error_variation"],
        default="naive_error_variation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method": f"{args.method}_train_quantile",
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "notes": (
            "train bands are tertiles of train-only reference values; "
            "val/test prompts and labels are copied unchanged"
        ),
        "datasets": [],
    }

    for dataset in args.datasets:
        manifest["datasets"].append(build_dataset(args.input_root, args.output_root, dataset, method=args.method))

    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
