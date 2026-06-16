from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Callable

import pandas as pd

from recipe.time_series_forecast import etth1_raw_dataset_builder as base_builder
from recipe.time_series_forecast.config_utils import get_dataset_lengths


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "datasets" / "RAW" / "ETT-small"
DEFAULT_DATASETS_RAW_ROOT = PROJECT_ROOT / "datasets" / "RAW"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "RL"

ETT_CSV_FILES = {
    "ETTH1": "ETTh1.csv",
    "ETTH2": "ETTh2.csv",
    "ETTM1": "ETTm1.csv",
    "ETTM2": "ETTm2.csv",
}

FULL_CSV_DATASETS = {
    "ETTH1": {
        "csv_path": Path("ETT-small/ETTh1.csv"),
        "field": "OT",
        "stride": 96,
    },
    "ETTH2": {
        "csv_path": Path("ETT-small/ETTh2.csv"),
        "field": "OT",
        "stride": 96,
    },
    "ETTM1": {
        "csv_path": Path("ETT-small/ETTm1.csv"),
        "field": "OT",
        "stride": 96,
    },
    "ETTM2": {
        "csv_path": Path("ETT-small/ETTm2.csv"),
        "field": "OT",
        "stride": 96,
    },
    "WIND": {
        "csv_path": Path("Wind/wind.csv"),
        "field": "target",
        "stride": 96,
    },
}

PRESPLIT_CSV_DATASETS = {
    "NP": {"dir": "NP", "prefix": "EPF_NP", "field": "OT", "stride": 48},
    "PJM": {"dir": "PJM", "prefix": "EPF_PJM", "field": "OT", "stride": 48},
    "BE": {"dir": "BE", "prefix": "EPF_BE", "field": "OT", "stride": 48},
    "DE": {"dir": "DE", "prefix": "EPF_DE", "field": "OT", "stride": 48},
    "FR": {"dir": "FR", "prefix": "EPF_FR", "field": "OT", "stride": 48},
}

TeacherMSEFn = Callable[[list[str], list[float], list[float], int], float]


def parse_dataset_names(raw_names: str) -> list[str]:
    if raw_names.strip().lower() == "all":
        return [*FULL_CSV_DATASETS, *PRESPLIT_CSV_DATASETS]

    dataset_names = []
    valid_dataset_names = {**FULL_CSV_DATASETS, **PRESPLIT_CSV_DATASETS}
    for raw_name in raw_names.split(","):
        dataset_name = raw_name.strip().upper()
        if not dataset_name:
            continue
        if dataset_name not in valid_dataset_names:
            valid_names = ", ".join(valid_dataset_names)
            raise ValueError(f"Unknown dataset '{raw_name}'. Valid names: {valid_names}")
        dataset_names.append(dataset_name)

    if not dataset_names:
        raise ValueError("No ETT datasets requested")
    return dataset_names


def output_dataset_dir_name(dataset_name: str) -> str:
    dataset_name = dataset_name.upper()
    return "Wind" if dataset_name == "WIND" else dataset_name


def _make_teacher_mse_fn(
    server_url: str,
    teacher_model_name: str,
    dataset_name: str | None = None,
) -> TeacherMSEFn:
    def teacher_mse_fn(
        timestamps: list[str],
        values: list[float],
        ground_truth: list[float],
        prediction_length: int,
    ) -> float:
        return base_builder.get_teacher_mse(
            timestamps,
            values,
            ground_truth,
            prediction_length,
            url=server_url,
            model_name=teacher_model_name,
            dataset_name=dataset_name,
        )

    return teacher_mse_fn


def build_entropy_banddrop_records(
    csv_path: str | Path,
    *,
    data_source: str,
    field: str = base_builder.FIELD,
    lookback_window: int = base_builder.LOOKBACK_WINDOW,
    forecast_horizon: int = base_builder.FORECAST_HORIZON,
    stride: int = base_builder.STRIDE,
) -> pd.DataFrame:
    window_records = base_builder._build_window_records(
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
        entropy = base_builder.permutation_entropy(record["content_values"])
        if math.isnan(entropy):
            continue

        annotated_records.append(
            {
                "data_source": record["data_source"],
                "agent_name": record["agent_name"],
                "prompt": record["prompt"],
                "ability": record["ability"],
                "reward_model": record["reward_model"],
                "source_index": record["source_index"],
                "entropy": float(entropy),
            }
        )

    annotated_df = pd.DataFrame(annotated_records)
    if annotated_df.empty:
        return annotated_df

    annotated_df["entropy_rank"] = annotated_df["entropy"].rank(pct=True, method="average")
    return annotated_df.loc[annotated_df["entropy_rank"] < base_builder.ENT_NOISE_Q].reset_index(drop=True).copy()


def build_internal_time_series_records(
    csv_path: str | Path,
    *,
    data_source: str,
    field: str = base_builder.FIELD,
    lookback_window: int = base_builder.LOOKBACK_WINDOW,
    forecast_horizon: int = base_builder.FORECAST_HORIZON,
    stride: int = base_builder.STRIDE,
) -> pd.DataFrame:
    return pd.DataFrame(
        base_builder._build_window_records(
            csv_path,
            data_source=data_source,
            field=field,
            lookback_window=lookback_window,
            forecast_horizon=forecast_horizon,
            stride=stride,
            include_internal_columns=True,
        )
    )


def filter_entropy_banddrop_records(records_df: pd.DataFrame) -> pd.DataFrame:
    if records_df.empty:
        return records_df.copy()

    annotated_df = records_df.copy()
    annotated_df["entropy"] = annotated_df["content_values"].map(base_builder.permutation_entropy)
    annotated_df = annotated_df.loc[~annotated_df["entropy"].map(math.isnan)].copy()
    if annotated_df.empty:
        return annotated_df

    annotated_df["entropy_rank"] = annotated_df["entropy"].rank(pct=True, method="average")
    return annotated_df.loc[annotated_df["entropy_rank"] < base_builder.ENT_NOISE_Q].reset_index(drop=True).copy()


def build_train_only_banddrop_split_frames(
    raw_split_frames: dict[str, pd.DataFrame],
    *,
    variant: str,
    teacher_mse_fn: TeacherMSEFn | None = None,
    forecast_horizon: int,
    dataset_name: str | None = None,
) -> dict[str, pd.DataFrame]:
    if variant == "raw":
        return raw_split_frames

    split_frames = {split_name: split_df.copy() for split_name, split_df in raw_split_frames.items()}
    train_df = split_frames["train"]
    if variant == "banddrop":
        split_frames["train"] = filter_entropy_banddrop_records(train_df)
    elif variant == "teacher_banddrop":
        teacher_mse_fn = teacher_mse_fn or _make_teacher_mse_fn(
            base_builder.MODEL_SERVER_URL,
            base_builder.TEACHER_MODEL_NAME,
            dataset_name,
        )
        annotated_df = train_df.copy()
        annotated_df["entropy"] = annotated_df["content_values"].map(base_builder.permutation_entropy)
        annotated_df = annotated_df.loc[~annotated_df["entropy"].map(math.isnan)].copy()
        annotated_df["teacher_mse"] = [
            teacher_mse_fn(
                row["content_timestamps"],
                row["content_values"],
                row["ground_truth_values"],
                forecast_horizon,
            )
            for _, row in annotated_df.iterrows()
        ]
        annotated_df["mse_rank"] = annotated_df["teacher_mse"].rank(pct=True, method="average")
        annotated_df["entropy_rank"] = annotated_df["entropy"].rank(pct=True, method="average")
        annotated_df["band"] = annotated_df.apply(base_builder.assign_band, axis=1)
        split_frames["train"] = base_builder.filter_band0_records(annotated_df)
    else:
        raise ValueError("variant must be 'banddrop', 'teacher_banddrop', or 'raw'")

    return split_frames


def build_ett_rl_split_frames(
    dataset_name: str,
    *,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    variant: str = "banddrop",
    teacher_mse_fn: TeacherMSEFn | None = None,
    server_url: str = base_builder.MODEL_SERVER_URL,
    teacher_model_name: str = base_builder.TEACHER_MODEL_NAME,
    field: str = base_builder.FIELD,
    lookback_window: int = base_builder.LOOKBACK_WINDOW,
    forecast_horizon: int = base_builder.FORECAST_HORIZON,
    stride: int = base_builder.STRIDE,
) -> dict[str, pd.DataFrame]:
    dataset_name = dataset_name.upper()
    if dataset_name not in ETT_CSV_FILES:
        valid_names = ", ".join(ETT_CSV_FILES)
        raise ValueError(f"Unknown ETT dataset '{dataset_name}'. Valid names: {valid_names}")

    csv_path = Path(raw_dir) / ETT_CSV_FILES[dataset_name]
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing source CSV: {csv_path}")

    if variant == "raw":
        records_df = base_builder.build_time_series_records(
            csv_path,
            data_source=dataset_name,
            field=field,
            lookback_window=lookback_window,
            forecast_horizon=forecast_horizon,
            stride=stride,
        )
        return base_builder.split_records_contiguously(records_df)

    raw_records_df = base_builder.build_time_series_records(
        csv_path,
        data_source=dataset_name,
        field=field,
        lookback_window=lookback_window,
        forecast_horizon=forecast_horizon,
        stride=stride,
    )
    raw_public_split_frames = base_builder._split_dataframe_contiguously(raw_records_df)
    records_df = build_internal_time_series_records(
        csv_path,
        data_source=dataset_name,
        field=field,
        lookback_window=lookback_window,
        forecast_horizon=forecast_horizon,
        stride=stride,
    )
    raw_split_frames = base_builder._split_dataframe_contiguously(records_df)
    split_frames = build_train_only_banddrop_split_frames(
        raw_split_frames,
        variant=variant,
        teacher_mse_fn=teacher_mse_fn
        or _make_teacher_mse_fn(server_url, teacher_model_name, dataset_name)
        if variant == "teacher_banddrop"
        else None,
        forecast_horizon=forecast_horizon,
        dataset_name=dataset_name,
    )
    split_frames["val"] = raw_public_split_frames["val"]
    split_frames["test"] = raw_public_split_frames["test"]
    return base_builder.finalize_split_frames(split_frames)


def build_full_csv_banddrop_split_frames(
    dataset_name: str,
    *,
    raw_root: str | Path = DEFAULT_DATASETS_RAW_ROOT,
    variant: str = "banddrop",
    teacher_mse_fn: TeacherMSEFn | None = None,
    server_url: str = base_builder.MODEL_SERVER_URL,
    teacher_model_name: str = base_builder.TEACHER_MODEL_NAME,
    lookback_window: int | None = None,
    forecast_horizon: int | None = None,
) -> dict[str, pd.DataFrame]:
    dataset_name = dataset_name.upper()
    if dataset_name not in FULL_CSV_DATASETS:
        valid_names = ", ".join(FULL_CSV_DATASETS)
        raise ValueError(f"Unknown full CSV dataset '{dataset_name}'. Valid names: {valid_names}")

    dataset_config = FULL_CSV_DATASETS[dataset_name]
    raw_root = Path(raw_root)
    csv_path = raw_root / dataset_config["csv_path"]
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing source CSV: {csv_path}")
    default_lookback, default_horizon = get_dataset_lengths(dataset_name)
    lookback_window = lookback_window or default_lookback
    forecast_horizon = forecast_horizon or default_horizon

    if variant == "raw":
        records_df = base_builder.build_time_series_records(
            csv_path,
            data_source=dataset_name,
            field=str(dataset_config["field"]),
            lookback_window=lookback_window,
            forecast_horizon=forecast_horizon,
            stride=int(dataset_config["stride"]),
        )
        return base_builder.split_records_contiguously(records_df)

    raw_records_df = base_builder.build_time_series_records(
        csv_path,
        data_source=dataset_name,
        field=str(dataset_config["field"]),
        lookback_window=lookback_window,
        forecast_horizon=forecast_horizon,
        stride=int(dataset_config["stride"]),
    )
    raw_public_split_frames = base_builder._split_dataframe_contiguously(raw_records_df)
    records_df = build_internal_time_series_records(
        csv_path,
        data_source=dataset_name,
        field=str(dataset_config["field"]),
        lookback_window=lookback_window,
        forecast_horizon=forecast_horizon,
        stride=int(dataset_config["stride"]),
    )
    raw_split_frames = base_builder._split_dataframe_contiguously(records_df)
    split_frames = build_train_only_banddrop_split_frames(
        raw_split_frames,
        variant=variant,
        teacher_mse_fn=teacher_mse_fn
        or _make_teacher_mse_fn(server_url, teacher_model_name, dataset_name)
        if variant == "teacher_banddrop"
        else None,
        forecast_horizon=forecast_horizon,
        dataset_name=dataset_name,
    )
    split_frames["val"] = raw_public_split_frames["val"]
    split_frames["test"] = raw_public_split_frames["test"]
    return base_builder.finalize_split_frames(split_frames)


def build_presplit_banddrop_split_frames(
    dataset_name: str,
    *,
    raw_root: str | Path = DEFAULT_DATASETS_RAW_ROOT,
    variant: str = "banddrop",
    teacher_mse_fn: TeacherMSEFn | None = None,
    server_url: str = base_builder.MODEL_SERVER_URL,
    teacher_model_name: str = base_builder.TEACHER_MODEL_NAME,
    lookback_window: int | None = None,
    forecast_horizon: int | None = None,
) -> dict[str, pd.DataFrame]:
    dataset_name = dataset_name.upper()
    if dataset_name not in PRESPLIT_CSV_DATASETS:
        valid_names = ", ".join(PRESPLIT_CSV_DATASETS)
        raise ValueError(f"Unknown presplit dataset '{dataset_name}'. Valid names: {valid_names}")

    dataset_config = PRESPLIT_CSV_DATASETS[dataset_name]
    default_lookback, default_horizon = get_dataset_lengths(dataset_name)
    lookback_window = lookback_window or default_lookback
    forecast_horizon = forecast_horizon or default_horizon
    split_frames = {}
    for split_name in base_builder.SPLIT_NAMES:
        csv_path = Path(raw_root) / str(dataset_config["dir"]) / f"{dataset_config['prefix']}_{split_name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing source CSV: {csv_path}")

        if variant == "raw" or split_name != "train":
            split_frames[split_name] = base_builder.build_time_series_records(
                csv_path,
                data_source=dataset_name,
                field=str(dataset_config["field"]),
                lookback_window=lookback_window,
                forecast_horizon=forecast_horizon,
                stride=int(dataset_config["stride"]),
            )
        else:
            raw_df = build_internal_time_series_records(
                csv_path,
                data_source=dataset_name,
                field=str(dataset_config["field"]),
                lookback_window=lookback_window,
                forecast_horizon=forecast_horizon,
                stride=int(dataset_config["stride"]),
            )
            if variant == "teacher_banddrop":
                split_frames[split_name] = build_train_only_banddrop_split_frames(
                    {"train": raw_df, "val": raw_df.iloc[0:0], "test": raw_df.iloc[0:0]},
                    variant=variant,
                    teacher_mse_fn=teacher_mse_fn
                    or _make_teacher_mse_fn(server_url, teacher_model_name, dataset_name),
                    forecast_horizon=forecast_horizon,
                    dataset_name=dataset_name,
                )["train"]
            elif variant == "banddrop":
                split_frames[split_name] = filter_entropy_banddrop_records(raw_df)
            else:
                raise ValueError("variant must be 'banddrop', 'teacher_banddrop', or 'raw'")

    return base_builder.finalize_split_frames(split_frames)


def build_dataset_split_frames(
    dataset_name: str,
    *,
    raw_root: str | Path = DEFAULT_DATASETS_RAW_ROOT,
    variant: str = "banddrop",
    teacher_mse_fn: TeacherMSEFn | None = None,
    server_url: str = base_builder.MODEL_SERVER_URL,
    teacher_model_name: str = base_builder.TEACHER_MODEL_NAME,
    lookback_window: int | None = None,
    forecast_horizon: int | None = None,
) -> dict[str, pd.DataFrame]:
    dataset_name = dataset_name.upper()
    if dataset_name in FULL_CSV_DATASETS:
        return build_full_csv_banddrop_split_frames(
            dataset_name,
            raw_root=raw_root,
            variant=variant,
            teacher_mse_fn=teacher_mse_fn,
            server_url=server_url,
            teacher_model_name=teacher_model_name,
            lookback_window=lookback_window,
            forecast_horizon=forecast_horizon,
        )
    if dataset_name in PRESPLIT_CSV_DATASETS:
        return build_presplit_banddrop_split_frames(
            dataset_name,
            raw_root=raw_root,
            variant=variant,
            teacher_mse_fn=teacher_mse_fn,
            server_url=server_url,
            teacher_model_name=teacher_model_name,
            lookback_window=lookback_window,
            forecast_horizon=forecast_horizon,
        )

    valid_names = ", ".join([*FULL_CSV_DATASETS, *PRESPLIT_CSV_DATASETS])
    raise ValueError(f"Unknown dataset '{dataset_name}'. Valid names: {valid_names}")


def build_and_save_ett_rl_dataset(
    dataset_name: str,
    *,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    variant: str = "banddrop",
    teacher_mse_fn: TeacherMSEFn | None = None,
    server_url: str = base_builder.MODEL_SERVER_URL,
    teacher_model_name: str = base_builder.TEACHER_MODEL_NAME,
    field: str = base_builder.FIELD,
    lookback_window: int = base_builder.LOOKBACK_WINDOW,
    forecast_horizon: int = base_builder.FORECAST_HORIZON,
    stride: int = base_builder.STRIDE,
) -> dict[str, Path]:
    dataset_name = dataset_name.upper()
    split_frames = build_ett_rl_split_frames(
        dataset_name,
        raw_dir=raw_dir,
        variant=variant,
        teacher_mse_fn=teacher_mse_fn,
        server_url=server_url,
        teacher_model_name=teacher_model_name,
        field=field,
        lookback_window=lookback_window,
        forecast_horizon=forecast_horizon,
        stride=stride,
    )
    return base_builder.save_split_dataframes(split_frames, Path(output_root) / dataset_name)


def build_and_save_dataset(
    dataset_name: str,
    *,
    raw_root: str | Path = DEFAULT_DATASETS_RAW_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    variant: str = "banddrop",
    teacher_mse_fn: TeacherMSEFn | None = None,
    server_url: str = base_builder.MODEL_SERVER_URL,
    teacher_model_name: str = base_builder.TEACHER_MODEL_NAME,
    lookback_window: int | None = None,
    forecast_horizon: int | None = None,
) -> dict[str, Path]:
    dataset_name = dataset_name.upper()
    split_frames = build_dataset_split_frames(
        dataset_name,
        raw_root=raw_root,
        variant=variant,
        teacher_mse_fn=teacher_mse_fn,
        server_url=server_url,
        teacher_model_name=teacher_model_name,
        lookback_window=lookback_window,
        forecast_horizon=forecast_horizon,
    )
    return base_builder.save_split_dataframes(split_frames, Path(output_root) / output_dataset_dir_name(dataset_name))


def _summarize_saved_paths(output_paths: dict[str, Path]) -> str:
    parts = []
    for split_name in base_builder.SPLIT_NAMES:
        output_path = output_paths[split_name]
        split_df = pd.read_parquet(output_path)
        band_counts = {}
        if len(split_df) > 0:
            band_counts = split_df["extra_info"].map(lambda item: item.get("band")).dropna().value_counts().sort_index()
            band_counts = {int(band): int(count) for band, count in band_counts.items()}
        if band_counts:
            parts.append(f"{split_name}={len(split_df)} bands={band_counts}")
        else:
            parts.append(f"{split_name}={len(split_df)}")
    return "; ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build ETT RL parquet datasets with the same base recipe as RL/ETTH1: "
            "window the OT series, optionally drop band0, then split train/val/test as 7:1:2."
        )
    )
    parser.add_argument("--datasets", default="all", help="Comma-separated ETT names, or 'all'.")
    parser.add_argument("--variant", choices=["banddrop", "teacher_banddrop", "raw"], default="banddrop")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_DATASETS_RAW_ROOT)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--server-url", default=base_builder.MODEL_SERVER_URL)
    parser.add_argument("--teacher-model-name", default=base_builder.TEACHER_MODEL_NAME)
    parser.add_argument("--lookback-window", type=int, default=None)
    parser.add_argument("--forecast-horizon", type=int, default=None)
    args = parser.parse_args()

    for dataset_name in parse_dataset_names(args.datasets):
        output_paths = build_and_save_dataset(
            dataset_name,
            raw_root=args.raw_root,
            output_root=args.output_root,
            variant=args.variant,
            server_url=args.server_url,
            teacher_model_name=args.teacher_model_name,
            lookback_window=args.lookback_window,
            forecast_horizon=args.forecast_horizon,
        )
        print(f"{dataset_name}: {_summarize_saved_paths(output_paths)}")
        for split_name, output_path in output_paths.items():
            print(f"  {split_name}: {output_path}")


if __name__ == "__main__":
    main()
