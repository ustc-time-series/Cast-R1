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
import asyncio
import copy
import json
import random
import logging
import os
import re
from typing import Any, Optional
from uuid import uuid4

from arft.agent_flow.agent_flow import AgentFlowBase, AgentFlowOutput, AgentFlowStep, register
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.tools.schemas import ToolResponse
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

from recipe.time_series_forecast.prompts import *
from recipe.time_series_forecast.utils import (
    parse_time_series_string,
    parse_time_series_to_dataframe,
    format_predictions_to_string,
    get_last_timestamp,
    predict_time_series_async,
    extract_basic_statistics,
    format_basic_statistics,
    extract_within_channel_dynamics,
    format_within_channel_dynamics,
    extract_forecast_residuals,
    format_forecast_residuals,
    extract_data_quality,
    format_data_quality,
    extract_event_summary,
    format_event_summary,
)
from recipe.time_series_forecast.reward import (
    compute_raw_relative_score,
    extract_values_from_time_series_string,
    extract_ground_truth_values,
)
import numpy as np

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

FEATURE_TOOL_NAMES = {
    "extract_basic_statistics",
    "extract_within_channel_dynamics",
    "extract_forecast_residuals",
    "extract_data_quality",
    "extract_event_summary",
}
PREDICT_MODEL_NAMES = {"chronos2", "arima", "patchtst", "itransformer"}
DEFAULT_PREDICT_MODEL_ORDER = ["patchtst", "itransformer", "arima", "chronos2"]


@register("time_series_forecast_agent")
class TimeSeriesForecastAgentFlow(AgentFlowBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history_analysis = []
        self.time_series_data = ""  # Raw string data
        self.messages = []
        self.steps = []
        
        # Parsed data cache
        self.timestamps = None
        self.values = None
        self.prediction_results = None
        self.final_answer = None

        # Feature extraction caches
        self.basic_statistics = None
        self.within_channel_dynamics = None
        self.forecast_residuals = None
        self.data_quality = None
        self.event_summary = None

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level TimeSeriesForecastAgentFlow initialization")

        # Initialize tools from config file
        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.max_steps = kwargs.get("max_steps", 5)
        cls.max_parallel_calls = kwargs.get("max_parallel_calls", 5)
        cls.lookback_window = kwargs.get("lookback_window", 96)
        cls.forecast_horizon = kwargs.get("forecast_horizon", 96)
        cls.auto_finalize_prediction = kwargs.get("auto_finalize_prediction", False)
        cls.allowed_predict_model_names = kwargs.get("allowed_predict_model_names")
        cls.preferred_predict_model = kwargs.get("preferred_predict_model")
        cls.tool_parser = ToolParser.get_tool_parser(config.actor_rollout_ref.rollout.multi_turn.format, cls.tokenizer)
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.tool_schemas = TIMESERIES_TOOL_SCHEMAS

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentFlowOutput:
        raw_prompt = list(kwargs["raw_prompt"])
        self.time_series_data = raw_prompt[0]["content"]
        self.messages = self._build_initial_messages()
        
        # Get ground_truth from reward_model field
        reward_model = kwargs.get("reward_model", {})
        ground_truth = reward_model.get("ground_truth", "") if isinstance(reward_model, dict) else ""
        dataset_name = self._get_dataset_name(kwargs)
        
        # Parse the input data once at the beginning
        try:
            self.timestamps, self.values = parse_time_series_string(self.time_series_data)
        except Exception as e:
            logger.error(f"Error parsing time series data: {e}")
            self.timestamps, self.values = [], []

        metrics = {}
        request_id = uuid4().hex[:8]
        
        # For logging last step

        Rnum = random.randint(0, 200)

        MSE_data = np.nan
        MAE_data = np.nan

        num_steps = 0
        while num_steps < self.max_steps:
            num_steps += 1

            messages = self._build_current_messages()
            sanitized_response_text = None

            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages,
                    tools=self._current_tool_schemas(),
                    add_generation_prompt=True,
                    tokenize=True,
                ),
            )
            prompt_ids = self._truncate_prompt_ids_for_rollout(prompt_ids)

            with simple_timer("generate_sequences", metrics):
                output = await self.server_manager.generate(
                    request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params
                )
            response_ids = output.token_ids[: self.response_length]
            
            # Decode response to check for final answer
            response_text = await self.loop.run_in_executor(None, self.tokenizer.decode, response_ids)

            if Rnum == 17:
                # Log last step's input/output
                print(f"\n[{request_id}] === (step {num_steps}) ===")
                print(f"[{request_id}] INPUT:\n{messages[1]['content']}")
                print(f"[{request_id}] OUTPUT:\n{response_text}")
                print(f"[{request_id}] ===============================\n")
            
            final_stage_tool_violation = (
                self.prediction_results is not None and self._contains_tool_call_markup(response_text)
            )
            if final_stage_tool_violation:
                reward_score = -0.5
                final_answer = None
                logger.warning("Tool call markup emitted after prediction stage; final-stage protocol violation")
            else:
                # Check if response contains final answer with <answer> tags
                final_answer = self._extract_final_answer(response_text)
            
            if final_answer:
                # Final answer detected - but first validate the workflow was followed
                workflow_valid, workflow_penalty, workflow_msg = self._validate_workflow_completion(final_answer)
                
                if not workflow_valid:
                    # Workflow not completed - apply penalty (reward hacking prevention)
                    reward_score = workflow_penalty
                    logger.warning(f"Workflow violation detected: {workflow_msg}. Penalty: {reward_score}")
                    # Don't accept this as final answer - force model to continue
                    # Set final_answer to None so the loop continues
                    final_answer = None
                else:
                    # Workflow completed - compute reward based on prediction accuracy
                    self.final_answer = final_answer
                    reward_score = self._compute_final_reward(
                        final_answer, ground_truth, dataset_name=dataset_name
                    )
                    sanitized_response_text = self._build_sanitized_export_response_text(response_text, final_answer)
                    logger.info(f"Final answer detected. Reward score: {reward_score}")
            elif not final_stage_tool_violation:
                # No final answer yet - process tool calls
                assistant_content, parsed_tool_calls = await self.tool_parser.extract_tool_calls(response_ids)
                protocol_penalty = self._compute_tool_protocol_penalty(response_text)
                parsed_tool_calls = parsed_tool_calls[:self.max_parallel_calls]
                tool_calls, skipped_stage_tool_calls = self._filter_tool_calls_for_current_stage(parsed_tool_calls)

                if tool_calls:
                    self._append_message(
                        "assistant",
                        self._build_assistant_history_message(
                            raw_response_text=response_text,
                            assistant_content=assistant_content,
                            tool_calls=tool_calls,
                        ),
                    )

                model_progress_made = False
                for tool_call in tool_calls:
                    before_state = self._workflow_state()
                    await self._execute_tool_call(tool_call, **kwargs)
                    model_progress_made = model_progress_made or self._workflow_state() != before_state

                if self._should_auto_feature_after_tool_markup_failure(response_text, tool_calls):
                    logger.warning(
                        "Malformed tool markup before feature analysis; auto-executing fallback extract_basic_statistics"
                    )
                    self._append_message("assistant", response_text)
                    before_state = self._workflow_state()
                    await self.extract_basic_statistics(**kwargs)
                    model_progress_made = model_progress_made or self._workflow_state() != before_state
                elif skipped_stage_tool_calls and self._should_auto_predict_after_prediction_stage_stall(
                    response_text, tool_calls
                ):
                    fallback_model = self._preferred_predict_model()
                    logger.info(
                        "Auto-executing %s prediction after out-of-stage tool call during prediction stage",
                        fallback_model,
                    )
                    self._append_message("assistant", response_text)
                    await self.predict(model_name=fallback_model, **kwargs)
                    self._append_message("tool", self._format_prediction_results())
                    model_progress_made = True
                elif self._should_auto_predict_after_tool_markup_failure(response_text, tool_calls):
                    fallback_model = self._preferred_predict_model()
                    logger.warning(
                        "Malformed tool markup after feature analysis; auto-executing fallback predict_time_series with %s",
                        fallback_model,
                    )
                    self._append_message("assistant", response_text)
                    await self.predict(model_name=fallback_model, **kwargs)
                    self._append_message("tool", self._format_prediction_results())
                    model_progress_made = True

                if self._should_auto_predict_after_no_progress(tool_calls, model_progress_made):
                    fallback_model = self._preferred_predict_model()
                    logger.info(
                        "Auto-executing %s prediction after repeated feature tool call",
                        fallback_model,
                    )
                    await self.predict(model_name=fallback_model, **kwargs)
                    self._append_message("tool", self._format_prediction_results())
                    model_progress_made = True

                if self._should_auto_predict_after_prediction_stage_stall(response_text, tool_calls):
                    fallback_model = self._preferred_predict_model()
                    logger.info(
                        "Auto-executing %s prediction after stalled predict-stage response",
                        fallback_model,
                    )
                    if response_text.strip():
                        self._append_message("assistant", response_text)
                    await self.predict(model_name=fallback_model, **kwargs)
                    self._append_message("tool", self._format_prediction_results())
                    model_progress_made = True

                auto_final_reward = self._auto_finalize_prediction_if_enabled(
                    ground_truth=ground_truth,
                    dataset_name=dataset_name,
                )
                if auto_final_reward is not None:
                    reward_score = auto_final_reward
                    final_answer = self.final_answer
                    sanitized_response_text = self._build_sanitized_export_response_text(
                        response_text,
                        self.final_answer,
                    )
                else:
                    reward_score = self._compute_intermediate_tool_reward(model_progress_made=model_progress_made)

                reward_score += protocol_penalty

            if (
                sanitized_response_text is None
                and num_steps >= self.max_steps
                and self.prediction_results is not None
            ):
                sanitized_response_text = self._build_sanitized_export_response_text(
                    response_text,
                    self.prediction_results,
                )

            step = AgentFlowStep(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
                reward_score=reward_score,
            )
            step.extra_fields["sanitized_response_text"] = sanitized_response_text
            step = await self._postprocess(step, **{**kwargs, "raw_prompt": [dict(message) for message in messages]})
            step.extra_fields['reward_extra_info'] = {'MSE': MSE_data, 'MAE': MAE_data}
            self.steps.append(step)
            
            # If final answer is detected, we can stop
            if final_answer:
                break

        
        metric_prediction_text = self._metric_prediction_text()
        if metric_prediction_text and ground_truth:
            try:
                MSE_data, MAE_data = self._compute_mse_mae_from_time_series_text(metric_prediction_text, ground_truth)
                if np.isfinite(MSE_data) and np.isfinite(MAE_data):
                    logger.info(f"Metrics - MSE: {MSE_data:.4f}, MAE: {MAE_data:.4f}")
            except Exception as e:
                logger.error(f"Error calculating MSE/MAE: {e}")

        self._apply_no_final_answer_penalty()

        # 确保所有样本都有一致的 extra_fields（使用 nan 便于后续用 nanmean 计算）
        self.steps[-1].extra_fields['reward_extra_info'] = {'MSE': MSE_data, 'MAE': MAE_data}

        return AgentFlowOutput(steps=self.steps, metrics=metrics)

    def _validate_workflow_completion(self, final_answer: str) -> tuple[bool, float, str]:
        """
        Validate that the workflow was properly completed before accepting final answer.
        Prevents reward hacking where model skips tools and copies input data.
        
        Returns:
            Tuple of (is_valid, penalty_score, message)
        """
        # Check 1: Must have called at least one feature extraction tool
        if not self._has_any_feature_analysis():
            return False, -0.5, "No feature extraction tools were called. You must analyze the data first."
        
        # Check 2: Must have called predict_time_series
        if self.prediction_results is None:
            return False, -0.5, "predict_time_series was not called. You must get model predictions first."

        answer_values = extract_values_from_time_series_string(final_answer)
        finite_answer_values = [value for value in answer_values if np.isfinite(value)]
        if len(finite_answer_values) < self.forecast_horizon:
            return (
                False,
                -0.5,
                f"Final answer must contain at least {self.forecast_horizon} finite forecast values.",
            )
        
        # Check 3: Check if answer is just copying the input data (reward hacking detection)
        if self._is_copying_input(final_answer):
            return False, -1.0, "Answer appears to copy input data. Predictions must be for FUTURE timestamps."
        
        return True, 0.0, "Workflow completed correctly"
    
    def _is_copying_input(self, final_answer: str) -> bool:
        """
        Detect if the model is copying input data instead of providing predictions.
        Checks if timestamps in answer overlap significantly with input timestamps.
        """
        if self.timestamps is None or len(self.timestamps) == 0:
            return False
        
        # Extract timestamps from the answer
        answer_timestamps = []
        lines = final_answer.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Try to extract timestamp (format: "2018-05-19 00:00:00 value")
            match = re.match(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', line)
            if match:
                answer_timestamps.append(match.group(1))
        
        if not answer_timestamps:
            return False
        
        # Convert input timestamps to strings for comparison
        input_ts_strings = set()
        for ts in self.timestamps:
            if hasattr(ts, 'strftime'):
                input_ts_strings.add(ts.strftime('%Y-%m-%d %H:%M:%S'))
            else:
                input_ts_strings.add(str(ts))
        
        # Count how many answer timestamps are in the input
        overlap_count = sum(1 for ts in answer_timestamps if ts in input_ts_strings)
        overlap_ratio = overlap_count / len(answer_timestamps) if answer_timestamps else 0
        
        # If more than 50% of answer timestamps are from input, it's likely copying
        if overlap_ratio > 0.5:
            logger.warning(f"Detected input copying: {overlap_ratio:.1%} of answer timestamps match input")
            return True
        
        return False

    def _extract_final_answer(self, response_text: str) -> Optional[str]:
        """
        Extract final answer from response text if <answer> tags are present.
        
        Args:
            response_text: The model's response text
            
        Returns:
            The content inside <answer> tags, or None if not found
        """
        if self._contains_tool_call_markup(response_text):
            return None

        match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _contains_tool_call_markup(self, response_text: str) -> bool:
        """Detect tool markup that is invalid after prediction results exist."""
        return bool(re.search(r'</?(?:tool_call|tools)\b', response_text or "", re.IGNORECASE))

    def _build_sanitized_export_response_text(self, response_text: str, final_answer: Optional[str]) -> Optional[str]:
        """Build a compact export-safe terminal response for SFT trajectory dumping."""
        answer_body = (final_answer or "").strip()
        if not answer_body:
            return None

        raw_text = (response_text or "").strip()
        think_body = None
        if not self._contains_tool_call_markup(raw_text) and re.search(r"<answer>.*?</answer>", raw_text, re.IGNORECASE | re.DOTALL):
            think_match = re.search(r"<think>(.*?)</think>", raw_text, re.IGNORECASE | re.DOTALL)
            if think_match:
                think_body = think_match.group(1).strip()

        if not think_body:
            think_body = "no justified global correction"

        return f"<think>{think_body}</think>\n<answer>\n{answer_body}\n</answer>"

    def _metric_prediction_text(self) -> Optional[str]:
        """Validation metrics should reflect the model's final answer, not tool fallback."""
        return self.final_answer

    def _compute_final_reward(
        self,
        final_answer: str,
        ground_truth: str,
        dataset_name: Optional[str] = None,
        *,
        include_reference_prediction: bool = True,
    ) -> float:
        """
        Compute reward score based on final prediction and ground truth.
        
        Args:
            final_answer: The predicted values as string
            ground_truth: The ground truth values as string
            
        Returns:
            Reward score
        """
        if not ground_truth:
            return 0.0
        
        try:
            extra_info = {"dataset_name": dataset_name}
            if include_reference_prediction:
                extra_info["reference_prediction"] = self.prediction_results

            score = compute_raw_relative_score(
                data_source=dataset_name or "time_series",
                solution_str=f"<answer>{final_answer}</answer>",
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            return score
        except Exception as e:
            logger.error(f"Error computing final reward: {e}")
            return -0.5

    def _auto_finalize_prediction_if_enabled(
        self,
        *,
        ground_truth: str,
        dataset_name: Optional[str] = None,
    ) -> Optional[float]:
        """Use tool predictions as the terminal answer for selection-only lowtool runs."""
        if not getattr(self, "auto_finalize_prediction", False):
            return None
        if self.final_answer is not None or self.prediction_results is None:
            return None

        self.final_answer = self.prediction_results
        return self._compute_final_reward(
            self.final_answer,
            ground_truth,
            dataset_name=dataset_name,
            include_reference_prediction=False,
        )

    def _compute_mse_mae_from_time_series_text(self, prediction_text: str, ground_truth: str) -> tuple[float, float]:
        """Compute raw-scale MSE/MAE from prediction and ground-truth time-series text."""
        pred_values = extract_values_from_time_series_string(prediction_text)
        gt_values = extract_ground_truth_values(ground_truth)

        if not pred_values or not gt_values:
            return np.nan, np.nan

        min_len = min(len(pred_values), len(gt_values))
        if min_len <= 0:
            return np.nan, np.nan

        pred_arr = np.array(pred_values[:min_len], dtype=float)
        gt_arr = np.array(gt_values[:min_len], dtype=float)

        return float(np.mean((pred_arr - gt_arr) ** 2)), float(np.mean(np.abs(pred_arr - gt_arr)))

    def _workflow_state(self) -> tuple[bool, bool, bool, bool, bool, bool]:
        """Return coarse workflow state flags used to distinguish progress from duplicate calls."""
        return (
            self.basic_statistics is not None,
            self.within_channel_dynamics is not None,
            self.forecast_residuals is not None,
            self.data_quality is not None,
            self.event_summary is not None,
            self.prediction_results is not None,
        )

    def _compute_intermediate_tool_reward(self, *, model_progress_made: bool) -> float:
        """Only reward tool calls that move the workflow to a new state."""
        return 0.02 if model_progress_made else 0.0

    def _allowed_predict_models(self) -> list[str]:
        _, allowed = normalize_predict_model_policy(
            getattr(self, "preferred_predict_model", None),
            getattr(self, "allowed_predict_model_names", None),
        )
        return allowed

    def _has_predict_model_policy(self) -> bool:
        return bool(getattr(self, "allowed_predict_model_names", None)) or bool(
            getattr(self, "preferred_predict_model", None)
        )

    def _preferred_predict_model(self) -> str:
        preferred, _ = normalize_predict_model_policy(
            getattr(self, "preferred_predict_model", None),
            getattr(self, "allowed_predict_model_names", None),
        )
        return preferred

    def _normalize_predict_model_name(self, model_name: Any) -> str:
        candidate = str(model_name or "").strip().lower()
        if not self._has_predict_model_policy():
            if candidate not in PREDICT_MODEL_NAMES:
                return "chronos2"
            return candidate

        allowed = self._allowed_predict_models()
        if candidate not in allowed:
            return self._preferred_predict_model()
        return candidate

    def _predict_tool_schema(self, schema: dict) -> dict:
        if not self._has_predict_model_policy():
            return schema

        schema_copy = copy.deepcopy(schema)
        allowed = self._allowed_predict_models()
        preferred = self._preferred_predict_model()
        model_field = (
            schema_copy.get("function", {})
            .get("parameters", {})
            .get("properties", {})
            .get("model_name")
        )
        if isinstance(model_field, dict):
            model_field["enum"] = allowed
            model_field["description"] = (
                f"Model to use. Prefer '{preferred}'. Allowed: {', '.join(allowed)}."
            )
        return schema_copy

    def _current_tool_schemas(self) -> list[dict] | None:
        """Expose only the tools needed for the current workflow stage."""
        if self.prediction_results is not None:
            return None
        if not self._has_any_feature_analysis():
            return [
                schema
                for schema in self.tool_schemas
                if schema.get("function", {}).get("name") in FEATURE_TOOL_NAMES
            ]
        # Once any feature analysis exists, force the next step into prediction so
        # short max-step configs cannot spend their remaining turns on more tools.
        return [
            self._predict_tool_schema(schema)
            for schema in self.tool_schemas
            if schema.get("function", {}).get("name") == "predict_time_series"
        ]

    def _filter_tool_calls_for_current_stage(self, tool_calls: list[Any]) -> tuple[list[Any], bool]:
        """
        Enforce the current stage's tool schema on parsed tool calls.

        The model may still emit tools that are not exposed in the current prompt stage.
        We must ignore those calls instead of executing them, otherwise the rollout can
        drift into repeated feature-analysis loops during the prediction stage.
        """
        if not tool_calls:
            return tool_calls, False

        if not getattr(self, "tool_schemas", None):
            return tool_calls, False

        schemas = self._current_tool_schemas()
        if schemas is None:
            return [], True

        allowed_tool_names = {
            schema.get("function", {}).get("name")
            for schema in schemas
            if schema.get("function", {}).get("name")
        }

        filtered: list[Any] = []
        skipped = False
        for tool_call in tool_calls:
            tool_name = getattr(tool_call, "name", None)
            if tool_name in allowed_tool_names:
                filtered.append(tool_call)
                continue
            if tool_name in PREDICT_MODEL_NAMES and "predict_time_series" in allowed_tool_names:
                filtered.append(tool_call)
                continue
            skipped = True
            logger.warning(
                "Ignoring out-of-stage tool call '%s'; allowed tools for current stage: %s",
                tool_name,
                sorted(allowed_tool_names),
            )

        return filtered, skipped

    def _should_auto_predict_after_no_progress(self, tool_calls: list[Any], model_progress_made: bool) -> bool:
        """
        If the model repeats feature extraction after features exist, advance to prediction without
        rewarding that duplicate action. This prevents max-step exhaustion while preserving the
        learning signal that duplicate tools are not useful.
        """
        if model_progress_made or not tool_calls:
            return False
        if not self._has_any_feature_analysis() or self.prediction_results is not None:
            return False
        return any(getattr(tool_call, "name", None) in FEATURE_TOOL_NAMES for tool_call in tool_calls)

    def _should_auto_feature_after_tool_markup_failure(self, response_text: str, tool_calls: list[Any]) -> bool:
        """
        If the initial feature stage emits tool markup that the parser cannot recover, salvage the
        step by executing a minimal feature extraction once instead of repeating Turn 1 forever.
        """
        if tool_calls:
            return False
        if self._has_any_feature_analysis() or self.prediction_results is not None:
            return False
        return self._contains_tool_call_markup(response_text)

    def _should_auto_predict_after_tool_markup_failure(self, response_text: str, tool_calls: list[Any]) -> bool:
        """
        If the model reaches prediction stage but only emits malformed tool markup, salvage the step by
        executing the preferred prediction model once instead of exhausting the trajectory with a -0.5.
        """
        if tool_calls:
            return False
        if not self._has_any_feature_analysis() or self.prediction_results is not None:
            return False
        return self._contains_tool_call_markup(response_text)

    def _should_auto_predict_after_prediction_stage_stall(self, response_text: str, tool_calls: list[Any]) -> bool:
        """
        For auto-finalize configs, salvage predict-stage stalls where the model emits neither a tool call
        nor a final answer after feature extraction. This directly targets blank/free-form responses that
        otherwise exhaust the trajectory without ever reaching prediction.
        """
        if tool_calls:
            return False
        if not getattr(self, "auto_finalize_prediction", False):
            return False
        if not self._has_any_feature_analysis() or self.prediction_results is not None:
            return False

        text = (response_text or "").strip()
        if not text:
            return True

        # Leave structured markup to the dedicated tool-markup/answer handlers.
        if re.search(r"</?(?:tool_call|tools|answer|think)\b", text, re.IGNORECASE):
            return False
        return True

    def _compute_tool_protocol_penalty(self, response_text: str) -> float:
        """
        Penalize tool responses that exceed the configured execution budget or mix ignored final answers
        with tool markup. This creates a learning signal for lowtool runs where only the first parsed
        tool call is executed.
        """
        if not response_text:
            return 0.0

        raw_tool_call_count = len(re.findall(r'<tool_call>', response_text, re.IGNORECASE))
        penalty = 0.0

        if raw_tool_call_count > getattr(self, "max_parallel_calls", 1):
            penalty -= 0.1
        if raw_tool_call_count > 0 and re.search(r'<answer>.*?</answer>', response_text, re.IGNORECASE | re.DOTALL):
            penalty -= 0.05

        return penalty

    def _serialize_tool_call_for_history(self, tool_call: Any) -> str:
        """Canonicalize a parsed tool call before appending it back into chat history."""
        arguments = getattr(tool_call, "arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError:
                pass

        tool_name = getattr(tool_call, "name", None)
        if tool_name == "predict_time_series" or tool_name in PREDICT_MODEL_NAMES:
            normalized_arguments = dict(arguments) if isinstance(arguments, dict) else {}
            normalized_arguments["model_name"] = self._normalize_predict_model_name(
                normalized_arguments.get("model_name", tool_name)
            )
            payload = {
                "name": "predict_time_series",
                "arguments": normalized_arguments,
            }
        else:
            payload = {
                "name": tool_name,
                "arguments": arguments if arguments is not None else {},
            }
        return f"<tool_call>\n{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n</tool_call>"

    def _build_assistant_history_message(
        self,
        *,
        raw_response_text: str,
        assistant_content: str,
        tool_calls: list[Any],
    ) -> str:
        """
        Keep assistant history compact by storing parser-cleaned content plus only the executed tool calls.
        This avoids carrying over long repeated tool-call blocks into the next-turn prompt.
        """
        if tool_calls:
            return "\n".join(self._serialize_tool_call_for_history(tool_call) for tool_call in tool_calls)

        cleaned_content = (assistant_content or "").strip()
        if cleaned_content:
            return cleaned_content
        return raw_response_text

    def _apply_no_final_answer_penalty(self) -> None:
        """Penalize trajectories that exhaust max steps without a valid final answer."""
        if self.final_answer is None and self.steps:
            self.steps[-1].reward_score = -0.5

    def _get_predict_model_name(self, tool_call: FunctionCall) -> str:
        model_name = None
        if hasattr(tool_call, "arguments") and tool_call.arguments:
            args = tool_call.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if isinstance(args, dict):
                model_name = args.get("model_name")
        return self._normalize_predict_model_name(model_name)

    async def _execute_tool_call(self, tool_call: FunctionCall, **kwargs) -> None:
        tool_name = getattr(tool_call, "name", None)
        if not hasattr(self, "history_analysis") or self.history_analysis is None:
            self.history_analysis = []
        history_len_before = len(self.history_analysis)

        if tool_name == "predict_time_series" or tool_name in PREDICT_MODEL_NAMES:
            # 强制工作流：必须先进行特征分析才能调用预测。
            if not self._has_any_feature_analysis():
                logger.warning("predict_time_series called without feature analysis, rejected")
                return
            model_name = tool_name if tool_name in PREDICT_MODEL_NAMES else self._get_predict_model_name(tool_call)
            await self.predict(model_name=model_name, **kwargs)
        elif tool_name == "extract_basic_statistics":
            await self.extract_basic_statistics(**kwargs)
        elif tool_name == "extract_within_channel_dynamics":
            await self.extract_within_channel_dynamics(**kwargs)
        elif tool_name == "extract_forecast_residuals":
            await self.extract_forecast_residuals(**kwargs)
        elif tool_name == "extract_data_quality":
            await self.extract_data_quality(**kwargs)
        elif tool_name == "extract_event_summary":
            await self.extract_event_summary(**kwargs)
        else:
            logger.warning(f"Unknown tool call ignored: {tool_name}")
            return

    def _truncate_time_series_data(self, data: str, head: int = 5, tail: int = 5) -> str:
        """
        Truncate time series data to show only first and last few rows.
        This saves tokens while preserving context about data format and range.
        """
        lines = data.strip().split('\n')
        if len(lines) <= head + tail:
            return data
        
        head_lines = lines[:head]
        tail_lines = lines[-tail:]
        omitted = len(lines) - head - tail
        
        return '\n'.join(head_lines) + f'\n... ({omitted} rows omitted) ...\n' + '\n'.join(tail_lines)

    def _get_current_turn_info(self) -> tuple[int, str]:
        """
        Determine current turn number and action based on state.
        Returns: (turn_number, action_instruction)
        """
        has_features = self._has_any_feature_analysis()
        has_predictions = self.prediction_results is not None
        
        if not has_features:
            return 1, "Call feature extraction tools (e.g., extract_basic_statistics). Do NOT call predict_time_series yet."
        elif not has_predictions:
            return 2, build_predict_stage_instruction(
                getattr(self, "preferred_predict_model", None),
                getattr(self, "allowed_predict_model_names", None),
            )
        else:
            return 3, (
                "Output your final answer in <think>...</think><answer>...</answer> format using the model predictions. "
                "Do NOT output any text outside these tags."
            )

    def _build_initial_user_prompt(self) -> str:
        """
        Build the initial user prompt for the first model turn.
        It must match the stateful replay/SFT prompt scaffold so the rollout
        starts from the same "Analysis History / Model Predictions" format
        seen during supervised training.
        """
        return self._build_user_prompt()

    def _build_initial_messages(self) -> list[dict]:
        return [
            {
                "role": "system",
                "content": build_timeseries_system_prompt(
                    getattr(self, "preferred_predict_model", None),
                    getattr(self, "allowed_predict_model_names", None),
                ),
            },
            {"role": "user", "content": self._build_initial_user_prompt()},
        ]

    def _build_user_prompt(self) -> str:
        """Backward-compatible dynamic prompt builder used by prompt/unit-test helpers."""
        history = self._format_history_analysis()
        prediction = self._format_prediction_results()
        turn_num, action = self._get_current_turn_info()
        use_truncated_history = self._has_any_feature_analysis() or self.prediction_results is not None
        historical_data_title = "### Historical Data (truncated)" if use_truncated_history else "### Historical Data"
        historical_data = (
            self._truncate_time_series_data(self.time_series_data) if use_truncated_history else self.time_series_data
        )

        if self.prediction_results:
            return f"""**[Turn {turn_num}] Action: {action}**
### Lookback Window: {self.lookback_window} rows
### Forecast Horizon: {self.forecast_horizon} rows
{historical_data_title}
{historical_data}

### Analysis History
{history}

### Model Predictions
{prediction}

**Instructions**:
1. Treat the Model Predictions as a reference forecast, not ground truth
2. Do not treat the reference forecast as the final answer automatically; decide whether one justified global correction is needed
3. <think> must be <= 40 tokens; if no single systematic bias is clear, say "no justified global correction"
4. Use only one global offset or scale correction when level, trend, residual, or quality signals show clear systematic bias
5. After Model Predictions appear, tool use is closed; never emit <tool_call> or <tools>
6. Output ONLY <think>...</think> and <answer>...</answer>, nothing else
7. The <answer> content must contain exactly {self.forecast_horizon} future timestamp-value rows
8. Each row must use format "YYYY-MM-DD HH:MM:SS finite plain decimal", not NaN/Inf/null/placeholders."""

        return f"""**[Turn {turn_num}] Action: {action}**
### Lookback Window: {self.lookback_window} rows
### Forecast Horizon: {self.forecast_horizon} rows
{historical_data_title}
{historical_data}

### Analysis History
{history}

### Model Predictions
{prediction}

**Check your current state and act accordingly:**
- If "Analysis History" is empty → Call feature extraction tools (e.g., extract_basic_statistics). Do NOT call predict_time_series yet.
{build_predict_stage_state_check_line(
    getattr(self, "preferred_predict_model", None),
    getattr(self, "allowed_predict_model_names", None),
)}
"""

    def _build_current_messages(self) -> list[dict]:
        """Build the exact messages for the current agent step."""
        if not self.messages:
            self.messages = self._build_initial_messages()
        elif self.messages[-1]["role"] != "user":
            self.messages.append({"role": "user", "content": self._build_user_prompt()})
        return [dict(message) for message in self.messages]

    def _append_message(self, role: str, content: str) -> None:
        if content:
            if not hasattr(self, "messages") or self.messages is None:
                self.messages = []
            self.messages.append({"role": role, "content": content})

    def _format_history_analysis(self) -> str:
        """Format history analysis records"""
        if not self.history_analysis:
            return "No previous analysis performed."
        
        return "\n".join(self.history_analysis)  # Show all analyses

    def _format_prediction_results(self) -> str:
        """Format prediction results for prompt"""
        if not self.prediction_results:
            return "No predictions available yet. Call predict_time_series to generate forecasts."
        
        return f"Model Predictions ({self.forecast_horizon} steps):\n{self.prediction_results}"

    def _has_any_feature_analysis(self) -> bool:
        """
        检查是否已经进行过任何特征分析。
        用于强制工作流：必须先进行特征分析，才能调用 predict_time_series。
        """
        return any([
            self.basic_statistics is not None,
            self.within_channel_dynamics is not None,
            self.forecast_residuals is not None,
            self.data_quality is not None,
            self.event_summary is not None,
        ])

    def _get_dataset_name(self, kwargs: dict[str, Any]) -> str | None:
        """Extract the RL dataset name from per-sample metadata."""
        value = kwargs.get("data_source")
        if value is None:
            extra_info = kwargs.get("extra_info")
            if isinstance(extra_info, dict):
                value = extra_info.get("dataset_name") or extra_info.get("data_source")

        if isinstance(value, (list, tuple)) and value:
            value = value[0]
        elif hasattr(value, "tolist"):
            value = value.tolist()
            if isinstance(value, list) and value:
                value = value[0]
        if isinstance(value, bytes):
            value = value.decode("utf-8")

        if value is None:
            return None

        dataset_name = str(value).strip().upper()
        return dataset_name or None

    async def predict(self, model_name: str = "chronos2", **kwargs) -> float:
        """
        Generate predictions using the specified model.

        Args:
            model_name: Name of the model to use ("chronos2" or "arima")

        Returns:
            Reward score (0.0 for tool execution, actual reward at final step)
        """
        try:
            model_name = self._normalize_predict_model_name(model_name)
            if not self.values or len(self.values) < 2:
                logger.warning("Insufficient data for prediction")
                return 0.0

            # Convert to DataFrame for prediction
            context_df = parse_time_series_to_dataframe(self.time_series_data)
            dataset_name = self._get_dataset_name(kwargs)

            # Generate predictions using the specified model
            pred_df = await predict_time_series_async(
                context_df,
                prediction_length=self.forecast_horizon,
                model_name=model_name,
                dataset_name=dataset_name,
            )

            # Get last timestamp for formatting
            last_ts = get_last_timestamp(self.time_series_data)

            # Format predictions as string
            self.prediction_results = format_predictions_to_string(pred_df, last_ts)

            # Record to history with model name
            dataset_suffix = f" for dataset {dataset_name}" if dataset_name else ""
            self.history_analysis.append(
                f"Model Prediction: Generated {self.forecast_horizon}-step forecast using {model_name.upper()} model{dataset_suffix}"
            )

            logger.info(f"Prediction completed using {model_name} model")

            return 0.0  # No intermediate reward

        except Exception as e:
            logger.error(f"Error in predict with {model_name}: {e}")
            self.history_analysis.append(f"Prediction failed ({model_name}): {str(e)}")
            return 0.0

    async def extract_basic_statistics(self, **kwargs) -> float:
        """Extract core statistical features from time series data."""
        try:
            if not self.values or len(self.values) < 2:
                logger.warning("Insufficient data for basic statistics extraction")
                return 0.0

            features = extract_basic_statistics(data=self.values)
            self.basic_statistics = features

            analysis_record = format_basic_statistics(features)
            self.history_analysis.append(analysis_record)

            logger.info("Basic statistics extraction completed")
            return 0.0
        except Exception as e:
            logger.error(f"Error in extract_basic_statistics: {e}")
            return 0.0

    async def extract_within_channel_dynamics(self, **kwargs) -> float:
        """Extract within-channel dynamics features from time series data."""
        try:
            if not self.values or len(self.values) < 2:
                logger.warning("Insufficient data for within-channel dynamics extraction")
                return 0.0

            features = extract_within_channel_dynamics(data=self.values)
            self.within_channel_dynamics = features

            analysis_record = format_within_channel_dynamics(features)
            self.history_analysis.append(analysis_record)

            logger.info("Within-channel dynamics extraction completed")
            return 0.0
        except Exception as e:
            logger.error(f"Error in extract_within_channel_dynamics: {e}")
            return 0.0

    async def extract_forecast_residuals(self, **kwargs) -> float:
        """Extract forecast residual features from time series data."""
        try:
            if not self.values or len(self.values) < 2:
                logger.warning("Insufficient data for forecast residuals extraction")
                return 0.0

            features = extract_forecast_residuals(data=self.values)
            self.forecast_residuals = features

            analysis_record = format_forecast_residuals(features)
            self.history_analysis.append(analysis_record)

            logger.info("Forecast residuals extraction completed")
            return 0.0
        except Exception as e:
            logger.error(f"Error in extract_forecast_residuals: {e}")
            return 0.0

    async def extract_data_quality(self, **kwargs) -> float:
        """Extract data quality features from time series data."""
        try:
            if not self.values or len(self.values) < 2:
                logger.warning("Insufficient data for data quality extraction")
                return 0.0

            features = extract_data_quality(data=self.values)
            self.data_quality = features

            analysis_record = format_data_quality(features)
            self.history_analysis.append(analysis_record)

            logger.info("Data quality extraction completed")
            return 0.0
        except Exception as e:
            logger.error(f"Error in extract_data_quality: {e}")
            return 0.0

    async def extract_event_summary(self, **kwargs) -> float:
        """Extract event summary features from time series data."""
        try:
            if not self.values or len(self.values) < 2:
                logger.warning("Insufficient data for event summary extraction")
                return 0.0

            features = extract_event_summary(data=self.values)
            self.event_summary = features

            analysis_record = format_event_summary(features)
            self.history_analysis.append(analysis_record)

            logger.info("Event summary extraction completed")
            return 0.0
        except Exception as e:
            logger.error(f"Error in extract_event_summary: {e}")
            return 0.0
