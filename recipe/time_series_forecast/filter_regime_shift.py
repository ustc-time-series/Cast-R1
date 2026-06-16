#!/usr/bin/env python3
"""
Filter test sets into regime-shift subsets for OOD evaluation.

For each dataset (NP / PJM / Wind), produces:
  - full            : all test samples (baseline reference)
  - high_volatility : top-K samples by MAD(ground_truth) / std(history)
  - changepoint     : top-K samples by #change-points in ground_truth
  - spike           : top-K samples by max |diff| in ground_truth

The filtering uses the **ground truth horizon** to identify hard windows,
which is valid for evaluation (the agent never sees ground truth at inference).

Outputs one JSON per dataset with sample indices for each subset.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recipe.time_series_forecast.reward import find_change_points
from recipe.time_series_forecast.utils import parse_time_series_string


def extract_values_plain(text: str) -> list[float]:
    """Extract float values from 'timestamp value' lines."""
    import re
    vals = []
    for line in text.strip().split("\n"):
        m = re.search(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+(-?\d+\.?\d*)', line)
        if m:
            vals.append(float(m.group(1)))
    return vals


def compute_mad(values: list[float]) -> float:
    arr = np.asarray(values)
    return float(np.mean(np.abs(arr - np.median(arr))))


def compute_changepoint_density(values: list[float]) -> int:
    if len(values) < 5:
        return 0
    maxs, mins = find_change_points(values)
    return len(maxs) + len(mins)


def compute_max_diff(values: list[float]) -> float:
    arr = np.asarray(values)
    return float(np.max(np.abs(np.diff(arr)))) if len(arr) > 1 else 0.0


def compute_relative_volatility(history: list[float], gt: list[float]) -> float:
    """MAD(gt) / (std(history) + eps)."""
    mad_gt = compute_mad(gt)
    std_hist = float(np.std(history)) + 1e-8
    return mad_gt / std_hist


def analyse_dataset(parquet_path: str, top_k: int = 20) -> dict:
    df = pd.read_parquet(parquet_path)
    records = []
    for idx, row in df.iterrows():
        try:
            hist_text = row["prompt"][0]["content"]
            gt_text = row["reward_model"]["ground_truth"]
        except Exception:
            continue
        _, hist_vals = parse_time_series_string(hist_text)
        gt_vals = extract_values_plain(gt_text)
        if not hist_vals or not gt_vals:
            continue
        records.append({
            "idx": int(idx),
            "mad_gt": compute_mad(gt_vals),
            "rel_vol": compute_relative_volatility(hist_vals, gt_vals),
            "n_cp": compute_changepoint_density(gt_vals),
            "max_diff": compute_max_diff(gt_vals),
            "hist_std": float(np.std(hist_vals)),
            "gt_range": float(np.max(gt_vals) - np.min(gt_vals)),
        })

    n = len(records)
    top_k = min(top_k, n // 3, n)  # don't take more than 1/3

    # Rank by each criterion
    by_rel_vol = sorted(records, key=lambda r: r["rel_vol"], reverse=True)
    by_cp = sorted(records, key=lambda r: r["n_cp"], reverse=True)
    by_spike = sorted(records, key=lambda r: r["max_diff"], reverse=True)

    subsets = {
        "full": {"indices": [r["idx"] for r in records], "count": n},
        "high_volatility": {
            "description": "Top-K by relative volatility (MAD_gt / std_hist)",
            "indices": [r["idx"] for r in by_rel_vol[:top_k]],
            "count": top_k,
            "stats": {
                "min_rel_vol": by_rel_vol[top_k - 1]["rel_vol"],
                "max_rel_vol": by_rel_vol[0]["rel_vol"],
                "mean_rel_vol": float(np.mean([r["rel_vol"] for r in by_rel_vol[:top_k]])),
            },
        },
        "changepoint_dense": {
            "description": "Top-K by #change-points in ground truth",
            "indices": [r["idx"] for r in by_cp[:top_k]],
            "count": top_k,
            "stats": {
                "min_cp": by_cp[top_k - 1]["n_cp"],
                "max_cp": by_cp[0]["n_cp"],
                "mean_cp": float(np.mean([r["n_cp"] for r in by_cp[:top_k]])),
            },
        },
        "spike": {
            "description": "Top-K by max single-step jump in ground truth",
            "indices": [r["idx"] for r in by_spike[:top_k]],
            "count": top_k,
            "stats": {
                "min_spike": by_spike[top_k - 1]["max_diff"],
                "max_spike": by_spike[0]["max_diff"],
                "mean_spike": float(np.mean([r["max_diff"] for r in by_spike[:top_k]])),
            },
        },
    }
    # dataset-level stats for context
    all_rel_vol = [r["rel_vol"] for r in records]
    all_cp = [r["n_cp"] for r in records]
    all_spike = [r["max_diff"] for r in records]
    subsets["dataset_stats"] = {
        "total_samples": n,
        "rel_vol_mean": float(np.mean(all_rel_vol)),
        "rel_vol_std": float(np.std(all_rel_vol)),
        "cp_mean": float(np.mean(all_cp)),
        "spike_mean": float(np.mean(all_spike)),
    }
    return subsets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=[
        "datasets/Wind/test.parquet",
        "datasets/NP/test.parquet",
        "datasets/PJM/test.parquet",
    ])
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--output-dir", default="benchmark_results/regime_shift")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds_path in args.datasets:
        name = Path(ds_path).parent.name.lower()
        print(f"\n=== {name} ({ds_path}) ===")
        result = analyse_dataset(ds_path, top_k=args.top_k)
        out_file = out_dir / f"{name}_subsets.json"
        out_file.write_text(json.dumps(result, indent=2))
        print(f"  total={result['full']['count']}")
        for k in ["high_volatility", "changepoint_dense", "spike"]:
            print(f"  {k}: n={result[k]['count']}, stats={result[k]['stats']}")
        print(f"  → {out_file}")


if __name__ == "__main__":
    main()
