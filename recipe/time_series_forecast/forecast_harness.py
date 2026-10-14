"""Shared tool execution and state construction for teacher, student and evaluation."""
from __future__ import annotations

import asyncio
import json

import numpy as np

from recipe.time_series_forecast.forecast_task import ForecastTask, build_state_messages, parse_rows, format_rows
from recipe.time_series_forecast.prompts import TIMESERIES_TOOL_SCHEMAS
from recipe.time_series_forecast import utils


DIAGNOSTICS = {
    "extract_basic_statistics": (utils.extract_basic_statistics, utils.format_basic_statistics, "basic_statistics"),
    "extract_within_channel_dynamics": (utils.extract_within_channel_dynamics, utils.format_within_channel_dynamics, "within_channel_dynamics"),
    "extract_forecast_residuals": (utils.extract_forecast_residuals, utils.format_forecast_residuals, "forecast_residuals"),
    "extract_data_quality": (utils.extract_data_quality, utils.format_data_quality, "data_quality"),
    "extract_event_summary": (utils.extract_event_summary, utils.format_event_summary, "event_summary"),
}


async def diagnose(task: ForecastTask, name: str) -> tuple[dict, str]:
    extract, formatter, _ = DIAGNOSTICS[name]
    if name == "extract_basic_statistics":
        data = np.column_stack([task.values, *task.historical_covariates.values()]).tolist()
        result = await asyncio.to_thread(extract, data=data)
        return result, formatter(result)
    result = await asyncio.to_thread(extract, data=task.values)
    sections = [f"Target {task.target_name}:\n{formatter(result)}"]
    for channel, values in task.historical_covariates.items():
        cov_result = await asyncio.to_thread(extract, data=values)
        sections.append(f"Covariate {channel}:\n{formatter(cov_result)}")
    return result, "\n".join(sections)


async def predict(task: ForecastTask, model_name: str) -> str:
    if model_name == "arima":
        context = utils.parse_time_series_to_dataframe(task.history_text)
        result = await utils.predict_with_arima_async(context, task.horizon)
        return utils.format_predictions_to_string(result)
    client = utils._get_httpx_client(utils._MODEL_SERVICE_URL)
    response = await client.post(f"{utils._MODEL_SERVICE_URL}/predict", json=task.model_request(model_name))
    response.raise_for_status()
    result = response.json()
    text = format_rows(result["timestamps"], result["values"])
    validate_forecast(task, text)
    return text


def validate_forecast(task: ForecastTask, text: str) -> list[float]:
    timestamps, values = parse_rows(text)
    if (len(values) != task.horizon or timestamps != task.future_timestamps or not np.isfinite(values).all()
            or len([line for line in text.splitlines() if line.strip()]) != task.horizon):
        raise ValueError("Forecast must cover the exact finite future timestamp grid")
    return values


class ForecastHarness:
    def __init__(self, task: ForecastTask):
        self.task = task
        self.analysis: list[str] = []
        self.prediction: str | None = None
        self.prediction_model: str | None = None
        self.tool_results: list[dict] = []

    def messages(self) -> list[dict]:
        return build_state_messages(self.task, self.analysis, self.prediction,
                                    prediction_model=self.prediction_model)

    def tools(self) -> list[dict]:
        allowed = set(DIAGNOSTICS) if not self.analysis else {"predict_time_series"}
        if self.prediction is not None:
            return []
        return [schema for schema in TIMESERIES_TOOL_SCHEMAS if schema["function"]["name"] in allowed]

    async def execute(self, calls: list[dict]) -> None:
        permitted = {schema["function"]["name"] for schema in self.tools()}
        if not calls or len(calls) > 5 or any(call["name"] not in permitted for call in calls):
            raise ValueError("Out-of-stage tool call")
        if self.analysis and len(calls) != 1:
            raise ValueError("The forecasting stage must select one forecasting model")
        async def execute_one(call):
            name = call["name"]
            if name in DIAGNOSTICS:
                _, observation = await diagnose(self.task, name)
            else:
                model = call.get("arguments", {}).get("model_name", "")
                if model not in {"arima", "chronos2", "patchtst", "itransformer"}:
                    raise ValueError(f"Unknown forecaster {model}")
                observation = await predict(self.task, model)
            return {"name": name, "arguments": call.get("arguments", {}), "observation": observation, "success": True}
        results = await asyncio.gather(*(execute_one(call) for call in calls))
        for result in results:
            self.tool_results.append(result)
            if result["name"] == "predict_time_series":
                self.prediction = result["observation"]
                self.prediction_model = result["arguments"]["model_name"]
            else:
                self.analysis.append(result["observation"])


def parse_tool_calls(text: str) -> list[dict]:
    import re
    return [json.loads(match) for match in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)]
