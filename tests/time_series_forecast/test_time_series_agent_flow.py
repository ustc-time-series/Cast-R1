import math
import asyncio
from types import SimpleNamespace

import pytest
import torch

from arft.agent_flow.agent_flow import _InternalAgentFlowStep
from recipe.time_series_forecast.prompts import TIMESERIES_SYSTEM_PROMPT
from recipe.time_series_forecast.time_series_forecast_agent_flow import TimeSeriesForecastAgentFlow
from recipe.time_series_forecast.utils import parse_time_series_string


def test_compute_mse_mae_from_time_series_text():
    flow = TimeSeriesForecastAgentFlow.__new__(TimeSeriesForecastAgentFlow)
    prediction = "\n".join(
        [
            "2020-01-01 00:00:00 1.0",
            "2020-01-01 01:00:00 3.0",
        ]
    )
    ground_truth = "\n".join(
        [
            "2020-01-01 00:00:00 1.0",
            "2020-01-01 01:00:00 1.0",
        ]
    )

    mse, mae = flow._compute_mse_mae_from_time_series_text(prediction, ground_truth)

    assert math.isfinite(mse)
    assert math.isfinite(mae)
    assert mse == pytest.approx(2.0)
    assert mae == pytest.approx(1.0)


def test_compute_mse_mae_from_time_series_text_returns_nan_without_predictions():
    flow = TimeSeriesForecastAgentFlow.__new__(TimeSeriesForecastAgentFlow)

    mse, mae = flow._compute_mse_mae_from_time_series_text("", "2020-01-01 00:00:00 1.0")

    assert math.isnan(mse)
    assert math.isnan(mae)


def test_intermediate_reward_only_rewards_state_changing_tool_progress():
    flow = TimeSeriesForecastAgentFlow.__new__(TimeSeriesForecastAgentFlow)

    assert flow._compute_intermediate_tool_reward(model_progress_made=True) > 0.0
    assert flow._compute_intermediate_tool_reward(model_progress_made=False) == 0.0


def test_repeated_feature_tool_can_trigger_auto_prediction_without_rewarding_model():
    flow = TimeSeriesForecastAgentFlow.__new__(TimeSeriesForecastAgentFlow)
    flow.basic_statistics = {"mean": 1.0}
    flow.within_channel_dynamics = None
    flow.forecast_residuals = None
    flow.data_quality = None
    flow.event_summary = None
    flow.prediction_results = None
    tool_calls = [SimpleNamespace(name="extract_basic_statistics")]

    assert flow._should_auto_predict_after_no_progress(tool_calls, model_progress_made=False)
    assert flow._compute_intermediate_tool_reward(model_progress_made=False) == 0.0


def test_max_steps_without_final_answer_penalizes_last_step():
    flow = TimeSeriesForecastAgentFlow.__new__(TimeSeriesForecastAgentFlow)
    flow.final_answer = None
    flow.steps = [SimpleNamespace(reward_score=0.02)]

    flow._apply_no_final_answer_penalty()

    assert flow.steps[-1].reward_score == pytest.approx(-0.5)


def test_tool_schemas_follow_workflow_state():
    flow = TimeSeriesForecastAgentFlow.__new__(TimeSeriesForecastAgentFlow)
    feature_schema = {"type": "function", "function": {"name": "extract_basic_statistics"}}
    second_feature_schema = {"type": "function", "function": {"name": "extract_within_channel_dynamics"}}
    predict_schema = {"type": "function", "function": {"name": "predict_time_series"}}
    flow.tool_schemas = [feature_schema, second_feature_schema, predict_schema]
    flow.basic_statistics = None
    flow.within_channel_dynamics = None
    flow.forecast_residuals = None
    flow.data_quality = None
    flow.event_summary = None
    flow.prediction_results = None

    assert flow._current_tool_schemas() == [feature_schema, second_feature_schema]

    flow.basic_statistics = {"mean": 1.0}
    assert flow._current_tool_schemas() == [predict_schema]

    flow.prediction_results = "2020-01-01 00:00:00 1.0"

    assert flow._current_tool_schemas() is None


def test_multiple_feature_tools_are_allowed_in_initial_feature_turn():
    flow = TimeSeriesForecastAgentFlow.__new__(TimeSeriesForecastAgentFlow)
    feature_schema = {"type": "function", "function": {"name": "extract_basic_statistics"}}
    second_feature_schema = {"type": "function", "function": {"name": "extract_within_channel_dynamics"}}
    predict_schema = {"type": "function", "function": {"name": "predict_time_series"}}
    flow.tool_schemas = [feature_schema, second_feature_schema, predict_schema]
    flow.basic_statistics = None
    flow.within_channel_dynamics = None
    flow.forecast_residuals = None
    flow.data_quality = None
    flow.event_summary = None
    flow.prediction_results = None

    tool_calls, skipped = flow._filter_tool_calls_for_current_stage(
        [SimpleNamespace(name="extract_basic_statistics"), SimpleNamespace(name="extract_within_channel_dynamics")]
    )

    assert [call.name for call in tool_calls] == ["extract_basic_statistics", "extract_within_channel_dynamics"]
    assert skipped is False


def test_later_feature_tool_is_filtered_out_after_initial_feature_turn():
    flow = TimeSeriesForecastAgentFlow.__new__(TimeSeriesForecastAgentFlow)
    feature_schema = {"type": "function", "function": {"name": "extract_basic_statistics"}}
    second_feature_schema = {"type": "function", "function": {"name": "extract_within_channel_dynamics"}}
    predict_schema = {"type": "function", "function": {"name": "predict_time_series"}}
    flow.tool_schemas = [feature_schema, second_feature_schema, predict_schema]
    flow.basic_statistics = {"mean": 1.0}
    flow.within_channel_dynamics = None
    flow.forecast_residuals = None
    flow.data_quality = None
    flow.event_summary = None
    flow.prediction_results = None

    tool_calls, skipped = flow._filter_tool_calls_for_current_stage(
        [SimpleNamespace(name="extract_within_channel_dynamics")]
    )

    assert tool_calls == []
    assert skipped is True


def test_workflow_rejects_final_answer_without_full_forecast_values():
    flow = TimeSeriesForecastAgentFlow.__new__(TimeSeriesForecastAgentFlow)
    flow.basic_statistics = {"mean": 1.0}
    flow.within_channel_dynamics = None
    flow.forecast_residuals = None
    flow.data_quality = None
    flow.event_summary = None
    flow.prediction_results = "2020-01-01 00:00:00 1.0\n2020-01-01 01:00:00 2.0"
    flow.timestamps = []
    flow.forecast_horizon = 2

    valid, penalty, message = flow._validate_workflow_completion("[use Model Predictions directly]")

    assert not valid
    assert penalty == pytest.approx(-0.5)
    assert "forecast values" in message


def test_workflow_rejects_final_stage_tool_call_markup():
    flow = TimeSeriesForecastAgentFlow.__new__(TimeSeriesForecastAgentFlow)

    assert flow._extract_final_answer(
        "<think>ready</think><tool_call>{}</tool_call><answer>\n2020-01-01 00:00:00 1.0\n</answer>"
    ) is None


def test_metric_prediction_text_requires_valid_final_answer():
    flow = TimeSeriesForecastAgentFlow.__new__(TimeSeriesForecastAgentFlow)
    flow.final_answer = None
    flow.prediction_results = "2020-01-01 00:00:00 9.0"

    assert flow._metric_prediction_text() is None

    flow.final_answer = "2020-01-01 00:00:00 1.0"
    assert flow._metric_prediction_text() == "2020-01-01 00:00:00 1.0"


def test_system_prompt_does_not_bias_toward_copying_predictions_by_default():
    assert "Copy the Model Predictions by default" not in TIMESERIES_SYSTEM_PROMPT
    assert "copy predictions" not in TIMESERIES_SYSTEM_PROMPT.lower()
    assert "reference forecast, not ground truth" in TIMESERIES_SYSTEM_PROMPT


def test_turn3_helper_prompt_does_not_instruct_default_copying():
    flow = TimeSeriesForecastAgentFlow.__new__(TimeSeriesForecastAgentFlow)
    flow.time_series_data = "2020-01-01 00:00:00 1.0"
    flow.lookback_window = 1
    flow.forecast_horizon = 1
    flow.history_analysis = ["Basic Statistics:\n  Median: 1.0"]
    flow.basic_statistics = {"median": 1.0}
    flow.within_channel_dynamics = None
    flow.forecast_residuals = None
    flow.data_quality = None
    flow.event_summary = None
    flow.prediction_results = "2020-01-02 00:00:00 1.5"

    prompt = flow._build_user_prompt()

    assert "Copy the Model Predictions into <answer> by default" not in prompt
    assert 'say "copy predictions"' not in prompt
    assert "reference forecast, not ground truth" in prompt


def test_initial_messages_use_stateful_user_prompt_matching_sft_replay_format():
    flow = TimeSeriesForecastAgentFlow.__new__(TimeSeriesForecastAgentFlow)
    flow.time_series_data = "2020-01-01 00:00:00 1.0"
    flow.lookback_window = 1
    flow.forecast_horizon = 1
    flow.history_analysis = []
    flow.basic_statistics = None
    flow.within_channel_dynamics = None
    flow.forecast_residuals = None
    flow.data_quality = None
    flow.event_summary = None
    flow.prediction_results = None

    messages = flow._build_initial_messages()

    assert messages[1]["content"] == flow._build_user_prompt()
    assert "### Analysis History" in messages[1]["content"]
    assert "No previous analysis performed." in messages[1]["content"]
    assert "### Model Predictions" in messages[1]["content"]
    assert "No predictions available yet." in messages[1]["content"]


def test_current_messages_append_fresh_user_prompt_after_prediction_tool_result():
    flow = TimeSeriesForecastAgentFlow.__new__(TimeSeriesForecastAgentFlow)
    flow.time_series_data = "2020-01-01 00:00:00 1.0"
    flow.lookback_window = 1
    flow.forecast_horizon = 1
    flow.history_analysis = ["Basic Statistics:\n  Median: 1.0"]
    flow.basic_statistics = {"median": 1.0}
    flow.within_channel_dynamics = None
    flow.forecast_residuals = None
    flow.data_quality = None
    flow.event_summary = None
    flow.prediction_results = "2020-01-02 00:00:00 1.5"
    flow.messages = flow._build_initial_messages()
    flow._append_message("assistant", '<tool_call>{"name":"predict_time_series","arguments":{"model_name":"itransformer"}}</tool_call>')
    flow._append_message("tool", flow._format_prediction_results())

    messages = flow._build_current_messages()

    assert messages[-1]["role"] == "user"
    assert "### Model Predictions" in messages[-1]["content"]
    assert "Output your final answer" in messages[-1]["content"]
    assert "Do NOT output any text outside these tags." in messages[-1]["content"]


def test_followup_turn2_prompt_truncates_repeated_historical_data():
    flow = TimeSeriesForecastAgentFlow.__new__(TimeSeriesForecastAgentFlow)
    flow.time_series_data = "\n".join(
        [f"2020-01-01 {hour:02d}:00:00 {hour}.0" for hour in range(12)]
    )
    flow.lookback_window = 12
    flow.forecast_horizon = 2
    flow.history_analysis = ["Basic Statistics:\n  Median: 5.5"]
    flow.basic_statistics = {"median": 5.5}
    flow.within_channel_dynamics = None
    flow.forecast_residuals = None
    flow.data_quality = None
    flow.event_summary = None
    flow.prediction_results = None

    prompt = flow._build_user_prompt()

    assert "### Historical Data (truncated)" in prompt
    assert "... (2 rows omitted) ..." in prompt
    assert "### Historical Data\n2020-01-01 00:00:00 0.0" not in prompt


def test_assistant_history_message_drops_long_reasoning_when_tool_calls_exist():
    flow = TimeSeriesForecastAgentFlow.__new__(TimeSeriesForecastAgentFlow)

    history_message = flow._build_assistant_history_message(
        raw_response_text="<think>very long reasoning</think><tool_call>{}</tool_call>",
        assistant_content="<think>very long reasoning</think>",
        tool_calls=[SimpleNamespace(name="extract_basic_statistics", arguments={})],
    )

    assert "very long reasoning" not in history_message
    assert '<tool_call>\n{"name":"extract_basic_statistics","arguments":{}}\n</tool_call>' in history_message


def test_execute_feature_tool_updates_state_without_replaying_tool_payload_message():
    async def run_call():
        flow = TimeSeriesForecastAgentFlow.__new__(TimeSeriesForecastAgentFlow)
        flow.time_series_data = "2020-01-01 00:00:00 1.0\n2020-01-01 01:00:00 2.0"
        flow.history_analysis = []
        flow.messages = [{"role": "assistant", "content": "before"}]
        flow.basic_statistics = None
        flow.within_channel_dynamics = None
        flow.forecast_residuals = None
        flow.data_quality = None
        flow.event_summary = None
        flow.prediction_results = None
        flow.timestamps, flow.values = parse_time_series_string(flow.time_series_data)

        await flow._execute_tool_call(SimpleNamespace(name="extract_basic_statistics", arguments={}))
        return flow

    flow = asyncio.run(run_call())

    assert flow.basic_statistics is not None
    assert flow.history_analysis
    assert [message["role"] for message in flow.messages] == ["assistant"]


def test_execute_predict_tool_updates_state_without_replaying_prediction_payload_message():
    async def run_call():
        flow = TimeSeriesForecastAgentFlow.__new__(TimeSeriesForecastAgentFlow)
        flow.time_series_data = "2020-01-01 00:00:00 1.0\n2020-01-01 01:00:00 2.0"
        flow.history_analysis = ["Basic Statistics:\n  Mean: 1.5"]
        flow.messages = [{"role": "assistant", "content": "before"}]
        flow.basic_statistics = {"mean": 1.5}
        flow.within_channel_dynamics = None
        flow.forecast_residuals = None
        flow.data_quality = None
        flow.event_summary = None
        flow.prediction_results = None
        flow.lookback_window = 2
        flow.forecast_horizon = 1
        flow.timestamps, flow.values = parse_time_series_string(flow.time_series_data)

        async def fake_predict(**kwargs):
            flow.prediction_results = "2020-01-01 02:00:00 3.0"
            return 0.0

        flow.predict = fake_predict

        await flow._execute_tool_call(SimpleNamespace(name="predict_time_series", arguments={"model_name": "patchtst"}))
        return flow

    flow = asyncio.run(run_call())

    assert flow.prediction_results == "2020-01-01 02:00:00 3.0"
    assert [message["role"] for message in flow.messages] == ["assistant"]


def test_policy_aware_turn2_prompt_uses_preferred_predict_model():
    flow = TimeSeriesForecastAgentFlow.__new__(TimeSeriesForecastAgentFlow)
    flow.time_series_data = "2020-01-01 00:00:00 1.0"
    flow.lookback_window = 1
    flow.forecast_horizon = 1
    flow.history_analysis = ["Basic Statistics:\n  Median: 1.0"]
    flow.basic_statistics = {"median": 1.0}
    flow.within_channel_dynamics = None
    flow.forecast_residuals = None
    flow.data_quality = None
    flow.event_summary = None
    flow.prediction_results = None
    flow.allowed_predict_model_names = ["itransformer"]
    flow.preferred_predict_model = "itransformer"

    prompt = flow._build_user_prompt()

    assert "itransformer" in prompt
    assert "chronos2" not in prompt


def test_policy_aware_initial_system_prompt_uses_preferred_predict_model():
    flow = TimeSeriesForecastAgentFlow.__new__(TimeSeriesForecastAgentFlow)
    flow.time_series_data = "2020-01-01 00:00:00 1.0"
    flow.lookback_window = 1
    flow.forecast_horizon = 1
    flow.history_analysis = []
    flow.basic_statistics = None
    flow.within_channel_dynamics = None
    flow.forecast_residuals = None
    flow.data_quality = None
    flow.event_summary = None
    flow.prediction_results = None
    flow.allowed_predict_model_names = ["itransformer"]
    flow.preferred_predict_model = "itransformer"

    messages = flow._build_initial_messages()

    assert "prefer 'itransformer'" in messages[0]["content"].lower()
    assert "prefer 'chronos2'" not in messages[0]["content"].lower()


def test_final_reward_penalizes_exact_tool_copy_when_worse_than_baseline():
    flow = TimeSeriesForecastAgentFlow.__new__(TimeSeriesForecastAgentFlow)
    tool_prediction = "\n".join(
        [
            "2020-01-01 00:00:00 7.0",
            "2020-01-01 01:00:00 7.0",
        ]
    )
    ground_truth = "\n".join(
        [
            "2020-01-01 00:00:00 0.0",
            "2020-01-01 01:00:00 0.0",
        ]
    )
    flow.prediction_results = tool_prediction

    reward = flow._compute_final_reward(tool_prediction, ground_truth, dataset_name="ETTH2")

    assert reward < 0.0


def test_final_reward_penalizes_exact_tool_copy_when_reference_is_good_but_not_perfect():
    flow = TimeSeriesForecastAgentFlow.__new__(TimeSeriesForecastAgentFlow)
    tool_prediction = "\n".join(
        [
            "2020-01-01 00:00:00 1.0",
            "2020-01-01 01:00:00 1.0",
        ]
    )
    ground_truth = "\n".join(
        [
            "2020-01-01 00:00:00 0.0",
            "2020-01-01 01:00:00 0.0",
        ]
    )
    flow.prediction_results = tool_prediction

    reward = flow._compute_final_reward(tool_prediction, ground_truth, dataset_name="ETTH1")

    assert reward < 0.0


def test_final_reward_allows_exact_perfect_tool_copy_to_remain_non_negative():
    flow = TimeSeriesForecastAgentFlow.__new__(TimeSeriesForecastAgentFlow)
    tool_prediction = "\n".join(
        [
            "2020-01-01 00:00:00 0.0",
            "2020-01-01 01:00:00 0.0",
        ]
    )
    ground_truth = tool_prediction
    flow.prediction_results = tool_prediction

    reward = flow._compute_final_reward(tool_prediction, ground_truth, dataset_name="ETTH1")

    assert reward >= 0.0


def test_auto_finalize_prediction_scores_tool_forecast_against_raw_baseline():
    flow = TimeSeriesForecastAgentFlow.__new__(TimeSeriesForecastAgentFlow)
    tool_prediction = "\n".join(
        [
            "2020-01-01 00:00:00 0.0",
            "2020-01-01 01:00:00 0.0",
        ]
    )
    flow.prediction_results = tool_prediction
    flow.final_answer = None
    flow.auto_finalize_prediction = True

    reward = flow._auto_finalize_prediction_if_enabled(
        ground_truth=tool_prediction,
        dataset_name="ETTH1",
    )

    assert flow.final_answer == tool_prediction
    assert reward > 0.0


def test_final_reward_prefers_raw_improvement_over_tool_prediction():
    flow = TimeSeriesForecastAgentFlow.__new__(TimeSeriesForecastAgentFlow)
    flow.prediction_results = "\n".join(
        [
            "2020-01-01 00:00:00 7.0",
            "2020-01-01 01:00:00 7.0",
        ]
    )
    improved_prediction = "\n".join(
        [
            "2020-01-01 00:00:00 1.0",
            "2020-01-01 01:00:00 1.0",
        ]
    )
    ground_truth = "\n".join(
        [
            "2020-01-01 00:00:00 0.0",
            "2020-01-01 01:00:00 0.0",
        ]
    )

    reward = flow._compute_final_reward(improved_prediction, ground_truth, dataset_name="ETTH2")

    assert reward > 0.0


def test_model_name_tool_call_is_treated_as_predict_tool():
    async def run_call():
        flow = TimeSeriesForecastAgentFlow.__new__(TimeSeriesForecastAgentFlow)
        flow.basic_statistics = {"mean": 1.0}
        flow.within_channel_dynamics = None
        flow.forecast_residuals = None
        flow.data_quality = None
        flow.event_summary = None
        flow.prediction_results = None
        captured = {}

        async def fake_predict(**kwargs):
            captured.update(kwargs)
            return 0.0

        flow.predict = fake_predict

        await flow._execute_tool_call(SimpleNamespace(name="patchtst", arguments={}))

        return captured

    assert asyncio.run(run_call())["model_name"] == "patchtst"


def test_predict_schema_can_be_restricted_to_allowed_models():
    flow = TimeSeriesForecastAgentFlow.__new__(TimeSeriesForecastAgentFlow)
    feature_schema = {"type": "function", "function": {"name": "extract_basic_statistics"}}
    predict_schema = {
        "type": "function",
        "function": {
            "name": "predict_time_series",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "enum": ["patchtst", "itransformer", "arima", "chronos2"],
                    }
                },
                "required": ["model_name"],
            },
        },
    }
    flow.tool_schemas = [feature_schema, predict_schema]
    flow.allowed_predict_model_names = ["itransformer"]
    flow.preferred_predict_model = "itransformer"
    flow.basic_statistics = {"mean": 1.0}
    flow.within_channel_dynamics = None
    flow.forecast_residuals = None
    flow.data_quality = None
    flow.event_summary = None
    flow.prediction_results = None

    tool_schemas = flow._current_tool_schemas()

    assert len(tool_schemas) == 1
    assert tool_schemas[0]["function"]["name"] == "predict_time_series"
    assert tool_schemas[0]["function"]["parameters"]["properties"]["model_name"]["enum"] == ["itransformer"]


def test_predict_tool_call_falls_back_to_allowed_preferred_model():
    async def run_call():
        flow = TimeSeriesForecastAgentFlow.__new__(TimeSeriesForecastAgentFlow)
        flow.basic_statistics = {"mean": 1.0}
        flow.within_channel_dynamics = None
        flow.forecast_residuals = None
        flow.data_quality = None
        flow.event_summary = None
        flow.prediction_results = None
        flow.allowed_predict_model_names = ["itransformer"]
        flow.preferred_predict_model = "itransformer"
        captured = {}

        async def fake_predict(**kwargs):
            captured.update(kwargs)
            return 0.0

        flow.predict = fake_predict

        await flow._execute_tool_call(
            SimpleNamespace(name="predict_time_series", arguments={"model_name": "chronos2"})
        )

        return captured

    assert asyncio.run(run_call())["model_name"] == "itransformer"


def test_run_salvages_malformed_initial_tool_markup_into_feature_stage_progress():
    malformed_tool_call = '<tool_call>\n{"name":"predict_time_series",""arguments":{}}\n</tool_call>'

    class DummyTokenizer:
        def apply_chat_template(self, messages, tools=None, add_generation_prompt=True, tokenize=True):
            return list(range(len(messages)))

        def decode(self, token_ids, skip_special_tokens=False):
            mapping = {
                (101,): malformed_tool_call,
                (102,): '<tool_call>{"name":"predict_time_series","arguments":{"model_name":"itransformer"}}</tool_call>',
                (103,): (
                    "<think>no justified global correction</think><answer>\n"
                    "2020-01-02 00:00:00 1.5\n"
                    "</answer>"
                ),
            }
            return mapping[tuple(token_ids)]

    class DummyServerManager:
        def __init__(self):
            self.calls = 0

        async def generate(self, request_id, prompt_ids, sampling_params):
            self.calls += 1
            token_ids = {
                1: [101],
                2: [102],
                3: [103],
            }[self.calls]
            return SimpleNamespace(token_ids=token_ids, log_probs=None)

    class DummyToolParser:
        async def extract_tool_calls(self, response_ids):
            mapping = {
                (101,): ("", []),
                (102,): ("", [SimpleNamespace(name="predict_time_series", arguments={"model_name": "itransformer"})]),
                (103,): ("", []),
            }
            return mapping[tuple(response_ids)]

    async def run_flow():
        flow = TimeSeriesForecastAgentFlow.__new__(TimeSeriesForecastAgentFlow)
        flow.config = SimpleNamespace(
            actor_rollout_ref=SimpleNamespace(
                rollout=SimpleNamespace(prompt_length=64, response_length=32),
            )
        )
        flow.loop = asyncio.get_running_loop()
        flow.tokenizer = DummyTokenizer()
        flow.processor = None
        flow.server_manager = DummyServerManager()
        flow.reward_manager_worker = None
        flow.tool_parser = DummyToolParser()
        flow.max_steps = 3
        flow.max_parallel_calls = 1
        flow.lookback_window = 2
        flow.forecast_horizon = 1
        flow.response_length = 32
        flow.auto_finalize_prediction = False
        flow.allowed_predict_model_names = ["itransformer"]
        flow.preferred_predict_model = "itransformer"
        flow.tool_schemas = [
            {"type": "function", "function": {"name": "extract_basic_statistics"}},
            {"type": "function", "function": {"name": "predict_time_series"}},
        ]
        flow.history_analysis = []
        flow.time_series_data = ""
        flow.messages = []
        flow.steps = []
        flow.timestamps = None
        flow.values = None
        flow.prediction_results = None
        flow.final_answer = None
        flow.basic_statistics = None
        flow.within_channel_dynamics = None
        flow.forecast_residuals = None
        flow.data_quality = None
        flow.event_summary = None

        async def fake_execute_tool_call(tool_call, **kwargs):
            if tool_call.name == "extract_basic_statistics":
                flow.basic_statistics = {"mean": 1.0}
                flow.history_analysis.append("Basic Statistics:\n  Mean: 1.0")
                return
            if tool_call.name == "predict_time_series":
                flow.prediction_results = "2020-01-02 00:00:00 1.5"
                return
            raise AssertionError(f"unexpected tool call: {tool_call.name}")

        async def fake_postprocess(step, **kwargs):
            step.extra_fields["raw_prompt"] = kwargs["raw_prompt"]
            prompt_ids = torch.tensor([step.prompt_ids], dtype=torch.long)
            response_ids = torch.tensor([step.response_ids], dtype=torch.long)
            input_ids = torch.cat([prompt_ids, response_ids], dim=1)
            attention_mask = torch.ones_like(input_ids)
            position_ids = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)
            response_mask = torch.ones_like(response_ids)
            return _InternalAgentFlowStep(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                response_mask=response_mask,
                reward_score=step.reward_score,
                num_turns=step.num_turns,
                extra_fields=step.extra_fields,
            )

        flow._execute_tool_call = fake_execute_tool_call
        flow._postprocess = fake_postprocess

        return await flow.run(
            sampling_params={},
            raw_prompt=[
                {
                    "role": "user",
                    "content": "2020-01-01 00:00:00 1.0\n2020-01-01 01:00:00 1.1",
                }
            ],
            reward_model={"ground_truth": "2020-01-02 00:00:00 1.5"},
            data_source="ETTH2",
        )

    output = asyncio.run(run_flow())

    step2_prompt = output.steps[1].extra_fields["raw_prompt"]

    assert [message["role"] for message in step2_prompt] == ["system", "user", "assistant", "user"]
    assert step2_prompt[2]["content"] == malformed_tool_call
    assert "Basic Statistics:" in step2_prompt[-1]["content"]
    assert "No previous analysis performed." not in step2_prompt[-1]["content"]
    assert "Call predict_time_series" in step2_prompt[-1]["content"]
    assert "No predictions available yet." in step2_prompt[-1]["content"]


def test_run_refreshes_user_prompt_between_feature_prediction_and_final_answer_turns():
    class DummyTokenizer:
        def apply_chat_template(self, messages, tools=None, add_generation_prompt=True, tokenize=True):
            return list(range(len(messages)))

        def decode(self, token_ids, skip_special_tokens=False):
            mapping = {
                (101,): '<tool_call>{"name":"extract_basic_statistics","arguments":{}}</tool_call>',
                (102,): '<tool_call>{"name":"predict_time_series","arguments":{"model_name":"itransformer"}}</tool_call>',
                (103,): (
                    "<think>no justified global correction</think><answer>\n"
                    "2020-01-02 00:00:00 1.5\n"
                    "</answer>"
                ),
            }
            return mapping[tuple(token_ids)]

    class DummyServerManager:
        def __init__(self):
            self.calls = 0

        async def generate(self, request_id, prompt_ids, sampling_params):
            self.calls += 1
            token_ids = {
                1: [101],
                2: [102],
                3: [103],
            }[self.calls]
            return SimpleNamespace(token_ids=token_ids, log_probs=None)

    class DummyToolParser:
        async def extract_tool_calls(self, response_ids):
            mapping = {
                (101,): ("", [SimpleNamespace(name="extract_basic_statistics", arguments={})]),
                (102,): ("", [SimpleNamespace(name="predict_time_series", arguments={"model_name": "itransformer"})]),
                (103,): ("", []),
            }
            return mapping[tuple(response_ids)]

    async def run_flow():
        flow = TimeSeriesForecastAgentFlow.__new__(TimeSeriesForecastAgentFlow)
        flow.config = SimpleNamespace(
            actor_rollout_ref=SimpleNamespace(
                rollout=SimpleNamespace(prompt_length=64, response_length=32),
            )
        )
        flow.loop = asyncio.get_running_loop()
        flow.tokenizer = DummyTokenizer()
        flow.processor = None
        flow.server_manager = DummyServerManager()
        flow.reward_manager_worker = None
        flow.tool_parser = DummyToolParser()
        flow.max_steps = 3
        flow.max_parallel_calls = 1
        flow.lookback_window = 2
        flow.forecast_horizon = 1
        flow.response_length = 32
        flow.auto_finalize_prediction = False
        flow.allowed_predict_model_names = ["itransformer"]
        flow.preferred_predict_model = "itransformer"
        flow.tool_schemas = [
            {"type": "function", "function": {"name": "extract_basic_statistics"}},
            {"type": "function", "function": {"name": "predict_time_series"}},
        ]
        flow.history_analysis = []
        flow.time_series_data = ""
        flow.messages = []
        flow.steps = []
        flow.timestamps = None
        flow.values = None
        flow.prediction_results = None
        flow.final_answer = None
        flow.basic_statistics = None
        flow.within_channel_dynamics = None
        flow.forecast_residuals = None
        flow.data_quality = None
        flow.event_summary = None

        async def fake_execute_tool_call(tool_call, **kwargs):
            if tool_call.name == "extract_basic_statistics":
                flow.basic_statistics = {"mean": 1.0}
                flow.history_analysis.append("Basic Statistics:\n  Mean: 1.0")
                flow._append_message("tool", flow.history_analysis[-1])
                return
            if tool_call.name == "predict_time_series":
                flow.prediction_results = "2020-01-02 00:00:00 1.5"
                flow._append_message("tool", flow._format_prediction_results())
                return
            raise AssertionError(f"unexpected tool call: {tool_call.name}")

        async def fake_postprocess(step, **kwargs):
            step.extra_fields["raw_prompt"] = kwargs["raw_prompt"]
            prompt_ids = torch.tensor([step.prompt_ids], dtype=torch.long)
            response_ids = torch.tensor([step.response_ids], dtype=torch.long)
            input_ids = torch.cat([prompt_ids, response_ids], dim=1)
            attention_mask = torch.ones_like(input_ids)
            position_ids = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)
            response_mask = torch.ones_like(response_ids)
            return _InternalAgentFlowStep(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                response_mask=response_mask,
                reward_score=step.reward_score,
                num_turns=step.num_turns,
                extra_fields=step.extra_fields,
            )

        flow._execute_tool_call = fake_execute_tool_call
        flow._postprocess = fake_postprocess

        return await flow.run(
            sampling_params={},
            raw_prompt=[
                {
                    "role": "user",
                    "content": "2020-01-01 00:00:00 1.0\n2020-01-01 01:00:00 1.1",
                }
            ],
            reward_model={"ground_truth": ""},
            data_source="ETTH2",
        )

    output = asyncio.run(run_flow())

    step2_prompt = output.steps[1].extra_fields["raw_prompt"]
    step3_prompt = output.steps[2].extra_fields["raw_prompt"]

    assert [message["role"] for message in step2_prompt] == ["system", "user", "assistant", "tool", "user"]
    assert "### Analysis History" in step2_prompt[-1]["content"]
    assert "Call predict_time_series" in step2_prompt[-1]["content"]
    assert "Model Predictions" in step2_prompt[-1]["content"]
    assert "No predictions available yet." in step2_prompt[-1]["content"]

    assert [message["role"] for message in step3_prompt] == [
        "system",
        "user",
        "assistant",
        "tool",
        "user",
        "assistant",
        "tool",
        "user",
    ]
    assert "### Model Predictions" in step3_prompt[-1]["content"]
    assert "Output your final answer" in step3_prompt[-1]["content"]
    assert "Do NOT output any text outside these tags." in step3_prompt[-1]["content"]
