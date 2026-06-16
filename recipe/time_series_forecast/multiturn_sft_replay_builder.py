import asyncio
import json
import math
import re
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

from recipe.time_series_forecast.prompts import (
    DEFAULT_ALLOWED_PREDICT_MODELS,
    build_predict_stage_state_check_line,
    build_timeseries_system_prompt,
    normalize_predict_model_policy,
)
from recipe.time_series_forecast.utils import (
    DEFAULT_FORECAST_HORIZON,
    DEFAULT_LOOKBACK_WINDOW,
    extract_basic_statistics,
    extract_data_quality,
    extract_event_summary,
    extract_forecast_residuals,
    extract_within_channel_dynamics,
    format_basic_statistics,
    format_data_quality,
    format_event_summary,
    format_forecast_residuals,
    format_predictions_to_string,
    format_within_channel_dynamics,
    get_last_timestamp,
    parse_time_series_string,
    parse_time_series_to_dataframe,
    predict_time_series_async,
)

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>.*?</answer>", re.DOTALL | re.IGNORECASE)
_ANSWER_BODY_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_TOOL_MARKUP_RE = re.compile(r"</?(?:tool_call|tools)\b", re.IGNORECASE)
_HISTORICAL_DATA_HEADER = "### Historical Data"
_PREDICT_MODEL_NAMES = set(DEFAULT_ALLOWED_PREDICT_MODELS)


def load_step_records(paths: Iterable[str | Path]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)

    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                grouped[record["trajectory_uid"]].append(record)

    ordered = OrderedDict()
    for trajectory_uid in sorted(grouped.keys()):
        ordered[trajectory_uid] = sorted(grouped[trajectory_uid], key=lambda record: record["step_index"])
    return ordered


def _finite_answer_value_count(response_text: str) -> int:
    match = _ANSWER_BODY_RE.search(response_text or "")
    if not match:
        return 0

    count = 0
    for line in match.group(1).strip().splitlines():
        value_match = re.search(r"(-?\d+(?:\.\d+)?)\s*$", line.strip())
        if not value_match:
            continue
        try:
            if math.isfinite(float(value_match.group(1))):
                count += 1
        except ValueError:
            continue
    return count


def has_valid_terminal_answer(records: list[dict], forecast_horizon: int | None = None) -> bool:
    if not records:
        return False

    terminal_record = max(records, key=lambda record: int(record.get("step_index", -1)))
    response_text = terminal_record.get("response_text", "") or ""
    if not bool(_ANSWER_RE.search(response_text)) or bool(_TOOL_MARKUP_RE.search(response_text)):
        return False
    if forecast_horizon is not None and _finite_answer_value_count(response_text) < forecast_horizon:
        return False
    return True


@dataclass
class ReplayState:
    time_series_data: str
    data_source: str | None = None
    lookback_window: int = DEFAULT_LOOKBACK_WINDOW
    forecast_horizon: int = DEFAULT_FORECAST_HORIZON
    history_analysis: list[str] = field(default_factory=list)
    prediction_results: str | None = None
    preferred_predict_model: str | None = None
    allowed_predict_model_names: list[str] = field(default_factory=list)

    def build_initial_messages(self) -> list[dict]:
        return [
            {
                "role": "system",
                "content": build_timeseries_system_prompt(
                    preferred_predict_model=self.preferred_predict_model,
                    allowed_predict_model_names=self.allowed_predict_model_names,
                ),
            },
            {"role": "user", "content": self._build_user_prompt()},
        ]

    def normalize_predict_model_name(self, model_name: Any) -> str:
        preferred, allowed = normalize_predict_model_policy(
            preferred_predict_model=self.preferred_predict_model,
            allowed_predict_model_names=self.allowed_predict_model_names,
        )
        candidate = str(model_name or "").strip().lower()
        if candidate not in allowed:
            return preferred
        return candidate

    def _build_user_prompt(self) -> str:
        history = self._format_history_analysis()
        prediction = self._format_prediction_results()
        return f"""**[Turn 1] Action: Call feature extraction tools (e.g., extract_basic_statistics). Do NOT call predict_time_series yet.**
### Lookback Window: {self.lookback_window} rows
### Forecast Horizon: {self.forecast_horizon} rows
### Historical Data
{self.time_series_data}

### Analysis History
{history}

### Model Predictions
{prediction}

**Check your current state and act accordingly:**
- If "Analysis History" is empty -> Call feature extraction tools (e.g., extract_basic_statistics). Do NOT call predict_time_series yet.
{build_predict_stage_state_check_line(
    preferred_predict_model=self.preferred_predict_model,
    allowed_predict_model_names=self.allowed_predict_model_names,
)}
"""

    def _format_history_analysis(self) -> str:
        if not self.history_analysis:
            return "No previous analysis performed."
        return "\n".join(self.history_analysis)

    def _format_prediction_results(self) -> str:
        if not self.prediction_results:
            return "No predictions available yet. Call predict_time_series to generate forecasts."
        return f"Model Predictions ({self.forecast_horizon} steps):\n{self.prediction_results}"


def _extract_time_series_data(raw_prompt: list[dict] | None) -> str:
    if not raw_prompt:
        raise ValueError("raw_prompt is required to rebuild the initial time series input")

    for message in reversed(raw_prompt):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if _HISTORICAL_DATA_HEADER in content:
            return _extract_section(content, _HISTORICAL_DATA_HEADER)
        return content.strip()

    raise ValueError("No user content found in raw_prompt")


def _extract_section(prompt_text: str, header: str) -> str:
    if header not in prompt_text:
        return prompt_text.strip()

    section = prompt_text.split(header, 1)[1].lstrip()
    next_header_index = section.find("\n### ")
    if next_header_index != -1:
        section = section[:next_header_index]
    return section.strip()


def _parse_tool_calls(response_text: str) -> list[dict[str, Any]]:
    tool_calls = []
    for payload in _TOOL_CALL_RE.findall(response_text or ""):
        payload = payload.strip()
        if not payload:
            continue
        tool_call = json.loads(payload)
        arguments = tool_call.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments) if arguments.strip() else {}
        tool_calls.append(
            {
                "name": tool_call.get("name"),
                "arguments": arguments if isinstance(arguments, dict) else {},
            }
        )
    return tool_calls


def _load_agent_predict_model_policy(agent_config_path: str | Path | None) -> tuple[str, list[str]]:
    if agent_config_path is None:
        return normalize_predict_model_policy()

    config_path = Path(agent_config_path)
    config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    candidates: list[dict[str, Any]] = []
    if isinstance(config_data, dict):
        candidates = [config_data]
    elif isinstance(config_data, list):
        candidates = [item for item in config_data if isinstance(item, dict)]

    selected = None
    for candidate in candidates:
        if candidate.get("name") == "time_series_forecast_agent":
            selected = candidate
            break
    if selected is None and candidates:
        selected = candidates[0]

    if not isinstance(selected, dict):
        return normalize_predict_model_policy()

    return normalize_predict_model_policy(
        preferred_predict_model=selected.get("preferred_predict_model"),
        allowed_predict_model_names=selected.get("allowed_predict_model_names"),
    )


def _canonicalize_tool_call(state: ReplayState, tool_name: str, arguments: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    normalized_arguments = dict(arguments) if isinstance(arguments, dict) else {}
    if tool_name == "predict_time_series" or tool_name in _PREDICT_MODEL_NAMES:
        normalized_arguments["model_name"] = state.normalize_predict_model_name(
            normalized_arguments.get("model_name", tool_name)
        )
        return "predict_time_series", normalized_arguments
    return tool_name, normalized_arguments


def _canonicalize_terminal_answer_message(response_text: str) -> str:
    text = (response_text or "").strip()
    answer_match = _ANSWER_BODY_RE.search(text)
    if not answer_match:
        return text

    parts: list[str] = []
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    if think_match:
        think_body = think_match.group(1).strip()
        parts.append(f"<think>{think_body}</think>")

    answer_body = answer_match.group(1).strip()
    if answer_body:
        parts.append(f"<answer>\n{answer_body}\n</answer>")
    else:
        parts.append("<answer>\n</answer>")

    return "\n".join(parts)


def _canonicalize_assistant_message(state: ReplayState, response_text: str) -> str:
    tool_calls = _parse_tool_calls(response_text)
    if not tool_calls:
        if _ANSWER_RE.search(response_text or ""):
            return _canonicalize_terminal_answer_message(response_text)
        return response_text

    assistant_content = _TOOL_CALL_RE.sub("", response_text or "").strip()
    parts: list[str] = []
    if assistant_content:
        parts.append(assistant_content)

    for tool_call in tool_calls:
        tool_name, arguments = _canonicalize_tool_call(state, tool_call["name"], tool_call["arguments"])
        parts.append(
            "<tool_call>\n"
            + json.dumps(
                {"name": tool_name, "arguments": arguments},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n</tool_call>"
        )

    return "\n".join(parts)


async def _execute_tool_call(state: ReplayState, tool_name: str, arguments: dict[str, Any]) -> str:
    _, values = parse_time_series_string(state.time_series_data)

    if tool_name == "extract_basic_statistics":
        observation = format_basic_statistics(extract_basic_statistics(values))
        state.history_analysis.append(observation)
        return observation

    if tool_name == "extract_within_channel_dynamics":
        observation = format_within_channel_dynamics(extract_within_channel_dynamics(values))
        state.history_analysis.append(observation)
        return observation

    if tool_name == "extract_forecast_residuals":
        observation = format_forecast_residuals(extract_forecast_residuals(values))
        state.history_analysis.append(observation)
        return observation

    if tool_name == "extract_data_quality":
        observation = format_data_quality(extract_data_quality(values))
        state.history_analysis.append(observation)
        return observation

    if tool_name == "extract_event_summary":
        observation = format_event_summary(extract_event_summary(values))
        state.history_analysis.append(observation)
        return observation

    if tool_name == "predict_time_series":
        model_name = state.normalize_predict_model_name(arguments.get("model_name"))
        context_df = parse_time_series_to_dataframe(state.time_series_data)
        pred_df = await predict_time_series_async(
            context_df,
            prediction_length=state.forecast_horizon,
            model_name=model_name,
            dataset_name=state.data_source,
        )
        last_ts = get_last_timestamp(state.time_series_data)
        state.prediction_results = format_predictions_to_string(pred_df, last_ts)
        state.history_analysis.append(
            f"Model Prediction: Generated {state.forecast_horizon}-step forecast using {model_name.upper()} model"
        )
        return f"Model Predictions ({state.forecast_horizon} steps):\n{state.prediction_results}"

    raise ValueError(f"Unsupported tool call in trajectory replay: {tool_name}")


async def trajectory_records_to_messages_async(
    records: list[dict],
    lookback_window: int = DEFAULT_LOOKBACK_WINDOW,
    forecast_horizon: int = DEFAULT_FORECAST_HORIZON,
    agent_config_path: str | Path | None = None,
) -> list[dict]:
    if not records:
        return []

    ordered_records = sorted(records, key=lambda record: record["step_index"])
    preferred_predict_model, allowed_predict_model_names = _load_agent_predict_model_policy(agent_config_path)
    state = ReplayState(
        time_series_data=_extract_time_series_data(ordered_records[0].get("raw_prompt")),
        data_source=ordered_records[0].get("data_source"),
        lookback_window=lookback_window,
        forecast_horizon=forecast_horizon,
        preferred_predict_model=preferred_predict_model,
        allowed_predict_model_names=allowed_predict_model_names,
    )
    messages = state.build_initial_messages()

    for record in ordered_records:
        response_text = record.get("response_text", "").strip()
        tool_calls = _parse_tool_calls(response_text)
        if response_text:
            messages.append({"role": "assistant", "content": _canonicalize_assistant_message(state, response_text)})

        for tool_call in tool_calls:
            tool_name, arguments = _canonicalize_tool_call(state, tool_call["name"], tool_call["arguments"])
            observation = await _execute_tool_call(state, tool_name, arguments)
            if observation:
                messages.append({"role": "tool", "content": observation})

    return messages


def trajectory_records_to_messages(
    records: list[dict],
    lookback_window: int = DEFAULT_LOOKBACK_WINDOW,
    forecast_horizon: int = DEFAULT_FORECAST_HORIZON,
    agent_config_path: str | Path | None = None,
) -> list[dict]:
    return asyncio.run(
        trajectory_records_to_messages_async(
            records,
            lookback_window=lookback_window,
            forecast_horizon=forecast_horizon,
            agent_config_path=agent_config_path,
        )
    )


def _score_value(row: dict) -> float:
    score = row.get("final_score")
    return float(score) if score is not None else float("-inf")


def _filter_rows_by_score(
    rows: list[dict],
    min_final_score: float | None = None,
    min_keep_count: int = 0,
) -> list[dict]:
    filtered = list(rows)
    if min_final_score is not None:
        filtered = [row for row in rows if _score_value(row) >= min_final_score]

    if min_keep_count > 0 and len(filtered) < min_keep_count:
        top_rows = sorted(rows, key=_score_value, reverse=True)[:min_keep_count]
        keep_ids = {row["trajectory_uid"] for row in top_rows}
        filtered = [row for row in rows if row["trajectory_uid"] in keep_ids]

    return filtered


async def build_multiturn_sft_rows_async(
    paths: Iterable[str | Path],
    lookback_window: int = DEFAULT_LOOKBACK_WINDOW,
    forecast_horizon: int = DEFAULT_FORECAST_HORIZON,
    min_final_score: float | None = None,
    min_keep_count: int = 0,
    agent_config_path: str | Path | None = None,
) -> list[dict]:
    grouped_records = load_step_records(paths)
    rows = []
    for trajectory_uid, steps in grouped_records.items():
        if not has_valid_terminal_answer(steps, forecast_horizon=forecast_horizon):
            continue

        rows.append(
            {
                "trajectory_uid": trajectory_uid,
                "sample_index": steps[0].get("sample_index"),
                "data_source": steps[0].get("data_source"),
                "rollout_n": steps[0].get("rollout_n"),
                "final_score": steps[0].get("final_score"),
                "ground_truth": steps[0].get("ground_truth"),
                "messages": await trajectory_records_to_messages_async(
                    steps,
                    lookback_window=lookback_window,
                    forecast_horizon=forecast_horizon,
                    agent_config_path=agent_config_path,
                ),
            }
        )

    rows = _filter_rows_by_score(rows, min_final_score=min_final_score, min_keep_count=min_keep_count)
    rows.sort(
        key=lambda row: (
            row["sample_index"] if row["sample_index"] is not None else -1,
            row["rollout_n"] if row["rollout_n"] is not None else -1,
        )
    )
    return rows


def build_multiturn_sft_rows(
    paths: Iterable[str | Path],
    lookback_window: int = DEFAULT_LOOKBACK_WINDOW,
    forecast_horizon: int = DEFAULT_FORECAST_HORIZON,
    min_final_score: float | None = None,
    min_keep_count: int = 0,
    agent_config_path: str | Path | None = None,
) -> list[dict]:
    return asyncio.run(
        build_multiturn_sft_rows_async(
            paths,
            lookback_window=lookback_window,
            forecast_horizon=forecast_horizon,
            min_final_score=min_final_score,
            min_keep_count=min_keep_count,
            agent_config_path=agent_config_path,
        )
    )


def write_multiturn_sft_parquet(
    paths: Iterable[str | Path],
    output_path: str | Path,
    lookback_window: int = DEFAULT_LOOKBACK_WINDOW,
    forecast_horizon: int = DEFAULT_FORECAST_HORIZON,
    min_final_score: float | None = None,
    min_keep_count: int = 0,
    agent_config_path: str | Path | None = None,
) -> None:
    rows = build_multiturn_sft_rows(
        paths,
        lookback_window=lookback_window,
        forecast_horizon=forecast_horizon,
        min_final_score=min_final_score,
        min_keep_count=min_keep_count,
        agent_config_path=agent_config_path,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Replay time-series rollout trajectories into verl multiturn SFT parquet."
    )
    parser.add_argument("--input_paths", nargs="+", required=True, help="One or more trajectory JSONL files.")
    parser.add_argument("--output_path", required=True, help="Output parquet path.")
    parser.add_argument("--lookback_window", type=int, default=DEFAULT_LOOKBACK_WINDOW)
    parser.add_argument("--forecast_horizon", type=int, default=DEFAULT_FORECAST_HORIZON)
    parser.add_argument("--min_final_score", type=float, default=None)
    parser.add_argument("--min_keep_count", type=int, default=0)
    parser.add_argument("--agent_config_path", default=None)
    args = parser.parse_args()

    write_multiturn_sft_parquet(
        args.input_paths,
        args.output_path,
        lookback_window=args.lookback_window,
        forecast_horizon=args.forecast_horizon,
        min_final_score=args.min_final_score,
        min_keep_count=args.min_keep_count,
        agent_config_path=args.agent_config_path,
    )
    print(f"Wrote replayed multiturn SFT parquet to {args.output_path}")


if __name__ == "__main__":
    main()
