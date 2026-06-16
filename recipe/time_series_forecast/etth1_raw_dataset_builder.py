from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import requests


LOOKBACK_WINDOW = 96
FORECAST_HORIZON = 96
STRIDE = 96
FIELD = "OT"
DATA_SOURCE = "ETTH1"
AGENT_NAME = "time_series_forecast_agent"
ABILITY = "TimeSeriesForecasting"
SPLIT_NAMES = ("train", "val", "test")
SPLIT_RATIOS = (7, 1, 2)
PHASE_NAMES = ("phase1", "phase2", "phase3")
PHASE_SIZE_RATIOS = (1, 1, 1)
PE_M = 3
PE_TAU = 1
ENT_LOW_Q = 0.50
ENT_NOISE_Q = 0.97
MSE_Q = 0.60
RNG_SEED = 42
MODEL_SERVER_URL = "http://127.0.0.1:8993/predict"
TEACHER_MODEL_NAME = "chronos2"

SCHEMA_COLUMNS = [
    "data_source",
    "agent_name",
    "prompt",
    "ability",
    "reward_model",
    "extra_info",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV_PATH = PROJECT_ROOT / "datasets" / "RAW" / "ETT-small" / "ETTh1.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "RAW" / "ETTH1"
DEFAULT_BANDDROP_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "RAW" / "ETTH1_BANDDROP"
DEFAULT_CURRICULUM_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "RAW" / "ETTH1_CURRICULUM"

TeacherMSEFn = Callable[[list[str], list[float], list[float], int], float]


def _format_value(value: float) -> str:
    return str(round(float(value), 3))


def _format_window(window_df: pd.DataFrame, field: str) -> str:
    return "\n".join(
        f"{timestamp} {_format_value(value)}"
        for timestamp, value in zip(window_df["date"], window_df[field], strict=False)
    )


def _allocate_counts(total: int, ratios: tuple[int | float, ...]) -> list[int]:
    if total <= 0:
        return [0 for _ in ratios]

    raw_counts = [total * ratio / sum(ratios) for ratio in ratios]
    counts = [int(count) for count in raw_counts]
    remainder = total - sum(counts)

    if remainder > 0:
        order = sorted(range(len(ratios)), key=lambda idx: raw_counts[idx] - counts[idx], reverse=True)
        for idx in order[:remainder]:
            counts[idx] += 1

    return counts


def _load_source_dataframe(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


def _build_window_records(
    csv_path: str | Path,
    *,
    data_source: str = DATA_SOURCE,
    field: str = FIELD,
    lookback_window: int = LOOKBACK_WINDOW,
    forecast_horizon: int = FORECAST_HORIZON,
    stride: int = STRIDE,
    include_internal_columns: bool = False,
) -> list[dict]:
    df = _load_source_dataframe(csv_path)

    records = []
    source_index = 0
    total_rows = len(df)
    for start_idx in range(0, total_rows, stride):
        content_end = start_idx + lookback_window
        ground_truth_end = content_end + forecast_horizon

        if ground_truth_end > total_rows:
            break

        content_df = df.iloc[start_idx:content_end]
        ground_truth_df = df.iloc[content_end:ground_truth_end]
        if content_df[field].isna().any() or ground_truth_df[field].isna().any():
            continue

        record = {
            "data_source": data_source,
            "agent_name": AGENT_NAME,
            "prompt": [{"content": _format_window(content_df, field), "role": "user"}],
            "ability": ABILITY,
            "reward_model": {"ground_truth": _format_window(ground_truth_df, field), "style": "rule"},
        }

        if include_internal_columns:
            record.update(
                {
                    "start_idx": start_idx,
                    "source_index": source_index,
                    "content_timestamps": content_df["date"].tolist(),
                    "content_values": content_df[field].astype(float).tolist(),
                    "ground_truth_values": ground_truth_df[field].astype(float).tolist(),
                }
            )

        records.append(record)
        source_index += 1

    return records


def build_time_series_records(
    csv_path: str | Path,
    *,
    data_source: str = DATA_SOURCE,
    field: str = FIELD,
    lookback_window: int = LOOKBACK_WINDOW,
    forecast_horizon: int = FORECAST_HORIZON,
    stride: int = STRIDE,
) -> pd.DataFrame:
    return pd.DataFrame(
        _build_window_records(
            csv_path,
            data_source=data_source,
            field=field,
            lookback_window=lookback_window,
            forecast_horizon=forecast_horizon,
            stride=stride,
            include_internal_columns=False,
        )
    )


def permutation_entropy(values: list[float], m: int = PE_M, tau: int = PE_TAU) -> float:
    values_array = np.asarray(values, dtype=float)
    num_patterns = len(values_array) - (m - 1) * tau
    if num_patterns <= 0:
        return float("nan")

    counts: dict[tuple[int, ...], int] = {}
    for idx in range(num_patterns):
        window = values_array[idx : idx + m * tau : tau]
        pattern = tuple(np.argsort(window))
        counts[pattern] = counts.get(pattern, 0) + 1

    probs = np.array(list(counts.values()), dtype=float)
    probs /= probs.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    return float(entropy / math.log(math.factorial(m)))


def get_teacher_mse(
    timestamps: list[str],
    values: list[float],
    ground_truth: list[float],
    prediction_length: int,
    *,
    url: str = MODEL_SERVER_URL,
    model_name: str = TEACHER_MODEL_NAME,
    dataset_name: str | None = None,
    timeout: int = 60,
    retries: int = 3,
) -> float:
    payload = {
        "timestamps": timestamps,
        "values": values,
        "model_name": model_name,
        "prediction_length": prediction_length,
    }
    if dataset_name:
        payload["dataset_name"] = dataset_name
    last_error: Exception | None = None
    for _ in range(retries):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            prediction = response.json()["values"]
            if len(prediction) != len(ground_truth):
                raise ValueError(f"prediction length {len(prediction)} != ground_truth {len(ground_truth)}")
            prediction_array = np.asarray(prediction, dtype=float)
            ground_truth_array = np.asarray(ground_truth, dtype=float)
            return float(np.mean((prediction_array - ground_truth_array) ** 2))
        except Exception as exc:  # pragma: no cover - exercised via runtime integration
            last_error = exc

    raise RuntimeError(f"teacher MSE failed after retries: {last_error}")


def assign_band(row: pd.Series) -> int:
    ent_rank = float(row["entropy_rank"])
    mse_rank = float(row["mse_rank"])

    if ent_rank >= ENT_NOISE_Q:
        return 0
    if ent_rank <= ENT_LOW_Q and mse_rank < MSE_Q:
        return 1
    if ent_rank <= ENT_LOW_Q and mse_rank >= MSE_Q:
        return 2
    if ent_rank > ENT_LOW_Q and mse_rank >= MSE_Q:
        return 3
    return 3


def build_annotated_time_series_records(
    csv_path: str | Path,
    *,
    teacher_mse_fn: TeacherMSEFn | None = None,
    data_source: str = DATA_SOURCE,
    field: str = FIELD,
    lookback_window: int = LOOKBACK_WINDOW,
    forecast_horizon: int = FORECAST_HORIZON,
    stride: int = STRIDE,
) -> pd.DataFrame:
    teacher_mse_fn = teacher_mse_fn or get_teacher_mse
    window_records = _build_window_records(
        csv_path,
        data_source=data_source,
        field=field,
        lookback_window=lookback_window,
        forecast_horizon=forecast_horizon,
        stride=stride,
        include_internal_columns=True,
    )

    annotated_records = []
    for record in window_records:
        entropy = permutation_entropy(record["content_values"])
        if math.isnan(entropy):
            continue

        teacher_mse = teacher_mse_fn(
            record["content_timestamps"],
            record["content_values"],
            record["ground_truth_values"],
            forecast_horizon,
        )

        annotated_records.append(
            {
                "data_source": record["data_source"],
                "agent_name": record["agent_name"],
                "prompt": record["prompt"],
                "ability": record["ability"],
                "reward_model": record["reward_model"],
                "start_idx": record["start_idx"],
                "source_index": record["source_index"],
                "entropy": float(entropy),
                "teacher_mse": float(teacher_mse),
            }
        )

    annotated_df = pd.DataFrame(annotated_records)
    if annotated_df.empty:
        return annotated_df

    annotated_df["mse_rank"] = annotated_df["teacher_mse"].rank(pct=True, method="average")
    annotated_df["entropy_rank"] = annotated_df["entropy"].rank(pct=True, method="average")
    annotated_df["band"] = annotated_df.apply(assign_band, axis=1)
    return annotated_df


def filter_band0_records(records_df: pd.DataFrame) -> pd.DataFrame:
    return records_df.loc[records_df["band"] != 0].reset_index(drop=True).copy()


def _split_dataframe_contiguously(
    records_df: pd.DataFrame,
    *,
    split_names: tuple[str, ...] = SPLIT_NAMES,
    split_ratios: tuple[int, ...] = SPLIT_RATIOS,
) -> dict[str, pd.DataFrame]:
    counts = _allocate_counts(len(records_df), split_ratios)
    split_frames: dict[str, pd.DataFrame] = {}

    start = 0
    for split_name, count in zip(split_names, counts, strict=False):
        split_frames[split_name] = records_df.iloc[start : start + count].copy().reset_index(drop=True)
        start += count

    return split_frames


def _build_extra_info(row: pd.Series, split_name: str, index: int) -> dict:
    extra_info: dict[str, int | str] = {"index": index, "split": split_name}
    if "band" in row and pd.notna(row["band"]):
        extra_info["band"] = int(row["band"])
    if "phase" in row and pd.notna(row["phase"]):
        extra_info["phase"] = str(row["phase"])
    if "source_index" in row and pd.notna(row["source_index"]):
        extra_info["source_index"] = int(row["source_index"])
    return extra_info


def finalize_split_frames(split_frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    finalized_frames: dict[str, pd.DataFrame] = {}
    for split_name, split_df in split_frames.items():
        split_df = split_df.copy().reset_index(drop=True)
        split_df["extra_info"] = [
            _build_extra_info(row, split_name=split_name, index=index)
            for index, (_, row) in enumerate(split_df.iterrows())
        ]
        finalized_frames[split_name] = split_df[SCHEMA_COLUMNS]
    return finalized_frames


def split_records_contiguously(
    records_df: pd.DataFrame,
    *,
    split_names: tuple[str, ...] = SPLIT_NAMES,
    split_ratios: tuple[int, ...] = SPLIT_RATIOS,
) -> dict[str, pd.DataFrame]:
    return finalize_split_frames(
        _split_dataframe_contiguously(records_df, split_names=split_names, split_ratios=split_ratios)
    )


def sample_pool(pool: pd.DataFrame, sample_size: int, rng: np.random.Generator) -> pd.DataFrame:
    if sample_size <= 0:
        return pool.iloc[0:0].copy()
    if len(pool) == 0:
        return pool.iloc[0:0].copy()
    random_state = int(rng.integers(0, 2**32 - 1))
    return pool.sample(n=sample_size, replace=True, random_state=random_state)


def build_phase(
    phase_name: str,
    phase_size: int,
    band1_pool: pd.DataFrame,
    band2_pool: pd.DataFrame,
    band3_pool: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    ratios = [
        0.75 if phase_name == "phase1" else 0.30 if phase_name == "phase2" else 0.20,
        0.25 if phase_name == "phase1" else 0.50 if phase_name == "phase2" else 0.40,
        0.00 if phase_name == "phase1" else 0.20 if phase_name == "phase2" else 0.40,
    ]
    pools = [band1_pool, band2_pool, band3_pool]
    available = [len(pool) > 0 for pool in pools]
    if not any(available):
        raise ValueError("all band pools are empty; check band thresholds")

    adjusted_ratios = [ratio if is_available else 0.0 for ratio, is_available in zip(ratios, available, strict=False)]
    total_ratio = sum(adjusted_ratios)
    adjusted_ratios = [ratio / total_ratio for ratio in adjusted_ratios]

    band_sizes = _allocate_counts(phase_size, tuple(adjusted_ratios))
    phase_parts = [
        sample_pool(band1_pool, band_sizes[0], rng),
        sample_pool(band2_pool, band_sizes[1], rng),
        sample_pool(band3_pool, band_sizes[2], rng),
    ]
    phase_df = pd.concat(phase_parts, ignore_index=True)
    phase_df["phase"] = phase_name
    return phase_df


def build_curriculum_train_df(train_base_df: pd.DataFrame, *, rng_seed: int = RNG_SEED) -> pd.DataFrame:
    if train_base_df.empty:
        curriculum_df = train_base_df.copy()
        curriculum_df["phase"] = []
        return curriculum_df

    rng = np.random.default_rng(rng_seed)
    phase_sizes = _allocate_counts(len(train_base_df), PHASE_SIZE_RATIOS)
    band1_pool = train_base_df.loc[train_base_df["band"] == 1].copy()
    band2_pool = train_base_df.loc[train_base_df["band"] == 2].copy()
    band3_pool = train_base_df.loc[train_base_df["band"] == 3].copy()

    phase_frames = [
        build_phase(phase_name, phase_size, band1_pool, band2_pool, band3_pool, rng)
        for phase_name, phase_size in zip(PHASE_NAMES, phase_sizes, strict=False)
    ]
    return pd.concat(phase_frames, ignore_index=True)


def save_split_dataframes(split_frames: dict[str, pd.DataFrame], output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: dict[str, Path] = {}
    for split_name, split_df in split_frames.items():
        output_path = output_dir / f"{split_name}.parquet"
        split_df.to_parquet(output_path, index=False)
        output_paths[split_name] = output_path

    return output_paths


def build_and_save_etth1_dataset(
    *,
    csv_path: str | Path = DEFAULT_CSV_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    records_df = build_time_series_records(csv_path)
    split_frames = split_records_contiguously(records_df)
    return save_split_dataframes(split_frames, output_dir)


def build_and_save_banddrop_etth1_dataset(
    *,
    csv_path: str | Path = DEFAULT_CSV_PATH,
    output_dir: str | Path = DEFAULT_BANDDROP_OUTPUT_DIR,
    teacher_mse_fn: TeacherMSEFn | None = None,
) -> dict[str, Path]:
    annotated_df = build_annotated_time_series_records(csv_path, teacher_mse_fn=teacher_mse_fn)
    filtered_df = filter_band0_records(annotated_df)
    split_frames = split_records_contiguously(filtered_df)
    return save_split_dataframes(split_frames, output_dir)


def build_and_save_curriculum_etth1_dataset(
    *,
    csv_path: str | Path = DEFAULT_CSV_PATH,
    output_dir: str | Path = DEFAULT_CURRICULUM_OUTPUT_DIR,
    teacher_mse_fn: TeacherMSEFn | None = None,
    rng_seed: int = RNG_SEED,
) -> dict[str, Path]:
    annotated_df = build_annotated_time_series_records(csv_path, teacher_mse_fn=teacher_mse_fn)
    filtered_df = filter_band0_records(annotated_df)
    base_split_frames = _split_dataframe_contiguously(filtered_df)
    split_frames = {
        "train": build_curriculum_train_df(base_split_frames["train"], rng_seed=rng_seed),
        "val": base_split_frames["val"],
        "test": base_split_frames["test"],
    }
    finalized_frames = finalize_split_frames(split_frames)
    return save_split_dataframes(finalized_frames, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ETTh1 parquet variants.")
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--variant", choices=["raw", "banddrop", "curriculum", "all"], default="raw")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--rng-seed", type=int, default=RNG_SEED)
    args = parser.parse_args()

    if args.variant == "raw":
        output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
        output_paths = build_and_save_etth1_dataset(csv_path=args.csv_path, output_dir=output_dir)
    elif args.variant == "banddrop":
        output_dir = args.output_dir or DEFAULT_BANDDROP_OUTPUT_DIR
        output_paths = build_and_save_banddrop_etth1_dataset(csv_path=args.csv_path, output_dir=output_dir)
    elif args.variant == "curriculum":
        output_dir = args.output_dir or DEFAULT_CURRICULUM_OUTPUT_DIR
        output_paths = build_and_save_curriculum_etth1_dataset(
            csv_path=args.csv_path,
            output_dir=output_dir,
            rng_seed=args.rng_seed,
        )
    else:
        raw_paths = build_and_save_etth1_dataset(csv_path=args.csv_path, output_dir=DEFAULT_OUTPUT_DIR)
        banddrop_paths = build_and_save_banddrop_etth1_dataset(
            csv_path=args.csv_path,
            output_dir=DEFAULT_BANDDROP_OUTPUT_DIR,
        )
        curriculum_paths = build_and_save_curriculum_etth1_dataset(
            csv_path=args.csv_path,
            output_dir=DEFAULT_CURRICULUM_OUTPUT_DIR,
            rng_seed=args.rng_seed,
        )
        output_paths = {
            **{f"raw_{name}": path for name, path in raw_paths.items()},
            **{f"banddrop_{name}": path for name, path in banddrop_paths.items()},
            **{f"curriculum_{name}": path for name, path in curriculum_paths.items()},
        }

    for split_name, output_path in output_paths.items():
        print(f"{split_name}: {output_path}")


if __name__ == "__main__":
    main()
