"""Validate the complete paper corpus before allocating GPU resources."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from recipe.time_series_forecast.config_utils import get_dataset_lengths, SHORT_FORECAST_DATASETS
from recipe.time_series_forecast.forecast_task import ForecastTask, parse_rows
from recipe.time_series_forecast.trajectory_distillation import DATASETS


def validate_corpus(root: Path, *, allow_missing_text: bool = False) -> dict:
    counts = {}
    for split in ("train", "val"):
        data = pd.read_parquet(root / f"{split}.parquet")
        bands, datasets = [], set()
        if data.empty:
            raise ValueError(f"Empty {split} corpus")
        for _, row in data.iterrows():
            info = row["extra_info"]
            if info.get("split") != split:
                raise ValueError(f"Wrong split in {split} file")
            task = ForecastTask.from_content(row["prompt"][0]["content"])
            dataset = task.dataset_name.upper()
            if dataset not in DATASETS or (len(task.values), task.horizon) != get_dataset_lengths(dataset):
                raise ValueError(f"Wrong paper task configuration for {dataset}")
            target_times, target_values = parse_rows(row["reward_model"]["ground_truth"])
            if len(target_values) != task.horizon or target_times != task.future_timestamps:
                raise ValueError(f"Target horizon/grid does not match the {dataset} input task")
            if not task.historical_covariates:
                raise ValueError(f"Missing historical covariates for {dataset}")
            if dataset in SHORT_FORECAST_DATASETS:
                if not task.future_covariates:
                    raise ValueError(f"Missing forecast-origin exogenous forecasts for {dataset}")
                if not task.text_context.strip() and not allow_missing_text:
                    raise ValueError(f"Missing original forecast-origin text for {dataset}; supply text/context columns")
            elif task.future_covariates or task.text_context:
                raise ValueError(f"LTSF task context must be empty for {dataset}")
            datasets.add(dataset)
            if split == "train":
                band = info.get("band")
                if band not in (1, 2, 3):
                    raise ValueError("Paper RL requires train-only entropy/reference-MSE bands")
                bands.append(band)
        if datasets != set(DATASETS):
            raise ValueError(f"{split} corpus is missing datasets: {sorted(set(DATASETS) - datasets)}")
        if bands != sorted(bands):
            raise ValueError("Training samples must follow the progressive easy-to-hard curriculum")
        counts[split] = len(data)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sft-model", type=Path)
    parser.add_argument("--allow-missing-text", action="store_true", help="Explicit context ablation, not the paper main experiment")
    args = parser.parse_args()
    counts = validate_corpus(args.data_root, allow_missing_text=args.allow_missing_text)
    if args.sft_model:
        manifest = json.loads((args.sft_model / "training_manifest.json").read_text())
        if manifest.get("num_trajectories") != 1200 or manifest.get("dataset_counts") != {name: 120 for name in DATASETS}:
            raise ValueError("The policy checkpoint must come from the balanced 1,200-trajectory SFT corpus")
        if not 7e9 <= manifest.get("num_parameters", 0) <= 9e9:
            raise ValueError("The paper main experiment requires the 8B policy")
        if manifest.get("model_type") != "qwen3":
            raise ValueError("The paper main experiment requires a Qwen3 policy")
        expected = {"precision": "bfloat16", "learning_rate": 1e-5, "epochs": 1,
                    "global_batch_size": 16, "prompt_length": 4096, "response_length": 2048}
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError("The SFT checkpoint manifest does not match the paper training settings")
    print(json.dumps({"validated": counts}))


if __name__ == "__main__":
    main()
