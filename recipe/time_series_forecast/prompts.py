# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

DEFAULT_ALLOWED_PREDICT_MODELS = ["patchtst", "itransformer", "arima", "chronos2"]
PREDICT_MODEL_DESCRIPTIONS = {
    "patchtst": "Strong local patterns, long-range dependencies",
    "itransformer": "Cross-channel correlations",
    "arima": "Clear trends, stable seasonality",
    "chronos2": "Default. Highly irregular data",
}


def normalize_predict_model_policy(
    preferred_predict_model: str | None = None,
    allowed_predict_model_names: list[str] | tuple[str, ...] | None = None,
) -> tuple[str, list[str]]:
    normalized: list[str] = []
    configured = allowed_predict_model_names or DEFAULT_ALLOWED_PREDICT_MODELS
    for model_name in configured:
        if not isinstance(model_name, str):
            continue
        candidate = model_name.strip().lower()
        if candidate in PREDICT_MODEL_DESCRIPTIONS and candidate not in normalized:
            normalized.append(candidate)

    if not normalized:
        normalized = list(DEFAULT_ALLOWED_PREDICT_MODELS)

    preferred = (
        preferred_predict_model.strip().lower()
        if isinstance(preferred_predict_model, str) and preferred_predict_model.strip()
        else None
    )
    if preferred not in normalized:
        preferred = normalized[0]

    return preferred, normalized


def build_timeseries_system_prompt(
    preferred_predict_model: str | None = None,
    allowed_predict_model_names: list[str] | tuple[str, ...] | None = None,
) -> str:
    preferred, allowed = normalize_predict_model_policy(
        preferred_predict_model=preferred_predict_model,
        allowed_predict_model_names=allowed_predict_model_names,
    )
    if len(allowed) == 1:
        turn2_instruction = (
            f"After seeing feature results in \"Analysis History\", call `predict_time_series` "
            f"with model_name '{preferred}'. Prefer '{preferred}'; it is the only allowed model."
        )
    else:
        model_lines = "\n".join(
            f"- '{model_name}': {PREDICT_MODEL_DESCRIPTIONS[model_name]}" for model_name in allowed
        )
        turn2_instruction = (
            f"After seeing feature results in \"Analysis History\", call `predict_time_series` "
            f"with chosen model (prefer '{preferred}' unless features strongly suggest otherwise):\n"
            f"{model_lines}"
        )

    return f"""You are a time series forecasting agent. This is a MULTI-TURN interaction.

## Workflow (MUST follow this order across turns)

**Turn 1 - Feature Extraction ONLY**:
Call one or more feature extraction tools. Do NOT call predict_time_series yet.
- `extract_basic_statistics`: median, MAD, autocorrelation, spectral features, correlation, PCA
- `extract_within_channel_dynamics`: changepoints, slopes, peaks, entropy
- `extract_forecast_residuals`: AR residual diagnostics
- `extract_data_quality`: quantization, saturation, dropout
- `extract_event_summary`: segment patterns (rise/fall/oscillation)

**Turn 2 - Prediction**:
{turn2_instruction}

**Turn 3 - Final Output**:
Treat the Model Predictions as a reference forecast, not ground truth. Do not treat the reference
forecast as the final answer automatically. Check it against the extracted evidence and produce an
independent final forecast. Apply at most one simple global offset or scale correction when the
extracted evidence shows a clear systematic bias.

## Output Format (Turn 3 only)
Your response MUST contain ONLY the two tags below, in this order, with no extra text before/between/after them.
Keep <think> to at most 40 tokens. If no single systematic bias is clear, say that there is no
justified global correction. Paste the actual predicted values inside <answer>.
<think>[<=40 tokens: no justified global correction, or name one global correction]</think>
<answer>
2017-05-05 00:00:00 12.345
...
</answer>

## CRITICAL RULES
- Turn 1: Feature extraction ONLY. Do NOT call predict_time_series.
- Turn 2: Call predict_time_series ONLY after features are extracted.
- Turn 3: Output answer ONLY after predictions are available.
- Do not emit a blind verbatim copy of the reference forecast by default.
- Either keep the reference forecast nearly unchanged because no justified global correction exists,
  or apply one global offset/scale backed by the extracted evidence.
- After Model Predictions appear, tool use is closed; never emit <tool_call> or <tools>.
- Each <answer> row must contain a future timestamp and one finite plain decimal value.
- Do NOT output anything outside <think> and <answer>. Missing <answer> tags is incorrect.
"""

TIMESERIES_SYSTEM_PROMPT = build_timeseries_system_prompt()


def build_predict_stage_instruction(
    preferred_predict_model: str | None = None,
    allowed_predict_model_names: list[str] | tuple[str, ...] | None = None,
) -> str:
    preferred, allowed = normalize_predict_model_policy(
        preferred_predict_model=preferred_predict_model,
        allowed_predict_model_names=allowed_predict_model_names,
    )
    if len(allowed) == 1:
        return f"Call predict_time_series with model_name '{preferred}'."
    return (
        "Call predict_time_series with your chosen model "
        f"(prefer '{preferred}' unless features strongly suggest another allowed model: {', '.join(allowed)})."
    )


def build_predict_stage_state_check_line(
    preferred_predict_model: str | None = None,
    allowed_predict_model_names: list[str] | tuple[str, ...] | None = None,
) -> str:
    preferred, allowed = normalize_predict_model_policy(
        preferred_predict_model=preferred_predict_model,
        allowed_predict_model_names=allowed_predict_model_names,
    )
    if len(allowed) == 1:
        return (
            '- If "Analysis History" has features but "Model Predictions" is empty '
            f"-> Call predict_time_series with model_name (must be '{preferred}')."
        )
    return (
        '- If "Analysis History" has features but "Model Predictions" is empty '
        f"-> Call predict_time_series with model_name (prefer '{preferred}' unless features strongly "
        f"suggest another allowed model: {', '.join(allowed)})."
    )


def build_predict_timeseries_tool_schema(
    preferred_predict_model: str | None = None,
    allowed_predict_model_names: list[str] | tuple[str, ...] | None = None,
) -> dict:
    preferred, allowed = normalize_predict_model_policy(
        preferred_predict_model=preferred_predict_model,
        allowed_predict_model_names=allowed_predict_model_names,
    )
    if len(allowed) == 1:
        model_summary = f"Model: '{preferred}'."
        model_description = f"Model to use. Must be '{preferred}'."
    else:
        model_summary = "Models: " + ", ".join(
            f"'{model_name}' ({PREDICT_MODEL_DESCRIPTIONS[model_name]})" for model_name in allowed
        ) + "."
        model_description = f"Model to use. Prefer '{preferred}' unless features strongly suggest another model."

    return {
        "type": "function",
        "function": {
            "name": "predict_time_series",
            "description": (
                "PREREQUISITE: You must have called feature extraction tools first "
                "(check 'Analysis History' is not empty). "
                "Do NOT call this on Turn 1 - extract features first! "
                f"{model_summary}"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": model_description,
                        "enum": allowed,
                    }
                },
                "required": ["model_name"],
            },
        },
    }


# OpenAI-compatible tool schemas for TimeSeriesForecast actions.
PREDICT_TIMESERIES_TOOL_SCHEMA = build_predict_timeseries_tool_schema()

EXTRACT_BASIC_STATISTICS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "extract_basic_statistics",
        "description": (
            "Extract core statistical features including median, MAD, autocorrelation, "
            "spectral features, CUSUM, quantile kurtosis, correlation, and PCA variance ratio."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

EXTRACT_WITHIN_CHANNEL_DYNAMICS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "extract_within_channel_dynamics",
        "description": (
            "Extract within-channel dynamics including changepoints, slopes, flatlines, "
            "peaks, entropy, and run-lengths."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

EXTRACT_FORECAST_RESIDUALS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "extract_forecast_residuals",
        "description": (
            "Extract AR residual diagnostics including mean, max, exceedance, ACF, "
            "and concentration."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

EXTRACT_DATA_QUALITY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "extract_data_quality",
        "description": (
            "Extract data quality metrics including quantization, saturation, "
            "constant channels, and dropout."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

EXTRACT_EVENT_SUMMARY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "extract_event_summary",
        "description": (
            "Extract event summary including segment count, rise/fall/flat/oscillation patterns."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

TIMESERIES_TOOL_SCHEMAS = [
    EXTRACT_BASIC_STATISTICS_SCHEMA,
    EXTRACT_WITHIN_CHANNEL_DYNAMICS_SCHEMA,
    EXTRACT_FORECAST_RESIDUALS_SCHEMA,
    EXTRACT_DATA_QUALITY_SCHEMA,
    EXTRACT_EVENT_SUMMARY_SCHEMA,
    PREDICT_TIMESERIES_TOOL_SCHEMA,
]
