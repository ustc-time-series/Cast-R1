"""Forecast-origin inputs shared by dataset construction, SFT, RL and inference.

Targets for the prediction window deliberately live outside this schema.
Legacy timestamp/value prompts remain readable for existing univariate datasets.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from recipe.time_series_forecast.prompts import (
    build_timeseries_system_prompt, build_predict_stage_instruction,
)


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
ROW = re.compile(rf"^(\d{{4}}-\d{{2}}-\d{{2}} \d{{2}}:\d{{2}}:\d{{2}})\s+({NUMBER})$")


def parse_rows(text: str) -> tuple[list[str], list[float]]:
    timestamps, values = [], []
    for line in str(text).strip().splitlines():
        match = ROW.fullmatch(line.strip())
        if match:
            timestamps.append(match[1])
            values.append(float(match[2]))
    return timestamps, values


def format_rows(timestamps: list[str], values: list[float]) -> str:
    return "\n".join(f"{ts} {float(value):.9g}" for ts, value in zip(timestamps, values, strict=True))


@dataclass
class ForecastTask:
    timestamps: list[str]
    values: list[float]
    horizon: int
    dataset_name: str = ""
    target_name: str = "target"
    historical_covariates: dict[str, list[float]] = field(default_factory=dict)
    future_covariates: dict[str, list[float]] = field(default_factory=dict)
    text_context: str = ""

    def __post_init__(self):
        if not self.values or len(self.timestamps) != len(self.values) or self.horizon <= 0:
            raise ValueError("A task requires aligned historical observations and a positive horizon")
        parsed = [datetime.fromisoformat(ts) for ts in self.timestamps]
        if len(parsed) > 1:
            delta = parsed[1] - parsed[0]
            if delta <= timedelta(0) or any(b - a != delta for a, b in zip(parsed, parsed[1:])):
                raise ValueError("Historical timestamps must form an increasing regular grid")
        if not all(math.isfinite(float(v)) for v in self.values):
            raise ValueError("Historical targets must be finite")
        for covariates, length in ((self.historical_covariates, len(self.values)), (self.future_covariates, self.horizon)):
            for name, values in covariates.items():
                if name in {self.target_name, "target", "ground_truth", "future_target"}:
                    raise ValueError("Future targets and labels cannot be supplied as covariates")
                if len(values) != length or not all(math.isfinite(float(v)) for v in values):
                    raise ValueError(f"Covariate {name!r} must have {length} finite values")
        if not set(self.future_covariates).issubset(self.historical_covariates):
            raise ValueError("Future covariates must also have historical context")

    @property
    def history_text(self) -> str:
        return format_rows(self.timestamps, self.values)

    @property
    def future_timestamps(self) -> list[str]:
        last = datetime.fromisoformat(self.timestamps[-1])
        delta = last - datetime.fromisoformat(self.timestamps[-2]) if len(self.timestamps) > 1 else timedelta(hours=1)
        return [(last + delta * i).strftime("%Y-%m-%d %H:%M:%S") for i in range(1, self.horizon + 1)]

    def to_dict(self) -> dict:
        return {
            "cast_r1_task": 1, "dataset_name": self.dataset_name, "target_name": self.target_name,
            "timestamps": self.timestamps, "values": self.values, "horizon": self.horizon,
            "historical_covariates": self.historical_covariates,
            "future_covariates": self.future_covariates, "text_context": self.text_context,
        }

    def serialize(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_content(cls, content: str, *, horizon: int = 96, dataset_name: str = "") -> ForecastTask:
        if str(content).lstrip().startswith("{"):
            data = json.loads(content)
            allowed = set(cls.__dataclass_fields__) | {"cast_r1_task"}
            if data.get("cast_r1_task") != 1 or set(data) - allowed:
                raise ValueError("Unsupported task schema or unexpected target/label fields")
            data.pop("cast_r1_task")
            return cls(**data)
        timestamps, values = parse_rows(content)
        return cls(timestamps=timestamps, values=values, horizon=horizon, dataset_name=dataset_name)

    def model_request(self, model_name: str) -> dict:
        return {
            "timestamps": self.timestamps, "values": self.values,
            "historical_covariates": self.historical_covariates,
            "future_covariates": self.future_covariates,
            "prediction_length": self.horizon, "model_name": model_name,
            "target_name": self.target_name,
            "dataset_name": self.dataset_name or None,
        }


def build_state_messages(task: ForecastTask, analysis: list[str], prediction: str | None,
                         preferred_model: str | None = None, allowed_models: list[str] | None = None,
                         prediction_model: str | None = None) -> list[dict]:
    """P(z, M): two fresh messages at every step, shared across training stages."""
    if prediction is not None:
        stage = "Refinement and Output"
        instruction = (
            "Treat model predictions as a reference forecast, not ground truth. "
            "Review and refine the forecast using recent observations, task context and memory. "
            "Keep predictions unchanged when no revision is justified. Negative electricity prices "
            "alone are not a reason to revise them. Output only <think>...</think><answer>...</answer> "
            f"with exactly {task.horizon} finite future timestamp-value rows. Do not call tools."
        )
        start = max(0, len(task.values) - 10)
    elif analysis:
        stage = "Forecasting"
        instruction = build_predict_stage_instruction(preferred_model, allowed_models)
        start = 0
    else:
        stage = "Preparation"
        instruction = "Call diagnostic tools only; do NOT call predict_time_series yet."
        start = 0
    interval = (datetime.fromisoformat(task.future_timestamps[0]) - datetime.fromisoformat(task.timestamps[-1])).total_seconds()
    context = {
        "historical_covariates": {k: v[start:] for k, v in task.historical_covariates.items()},
        "forecast_start": task.future_timestamps[0],
        "time_step_seconds": interval,
        "known_future_covariates": task.future_covariates,
        "text_available_at_forecast_origin": task.text_context,
    }
    prompt = (
        f"### Stage: {stage}\n### Lookback Window: {len(task.values)} rows\n"
        f"### Forecast Horizon: {task.horizon} rows\n### Target: {task.target_name}\n"
        f"### Historical Data{' (recent)' if start else ''}\n"
        f"Regular timestamp grid: {task.timestamps[start]} through {task.timestamps[-1]}, every {interval:g} seconds.\n"
        f"Target values in chronological order: {json.dumps(task.values[start:], separators=(',', ':'))}\n\n"
        f"### Task Context\n{json.dumps(context, ensure_ascii=False, allow_nan=False, separators=(',', ':'))}\n\n"
        f"### Analysis History\n{chr(10).join(analysis) if analysis else 'No previous analysis performed.'}\n\n"
        f"### Model Predictions{' (' + prediction_model + ')' if prediction_model else ''}\n"
        f"{prediction if prediction is not None else 'No predictions available yet.'}\n\n"
        f"### Permitted Action\n{instruction}"
    )
    return [{"role": "system", "content": build_timeseries_system_prompt(preferred_model, allowed_models)},
            {"role": "user", "content": prompt}]
