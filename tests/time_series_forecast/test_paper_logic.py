import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from arft.core_algos import compute_grpo_outcome_advantage
from recipe.time_series_forecast.forecast_task import ForecastTask, build_state_messages, format_rows
from recipe.time_series_forecast import etth1_raw_dataset_builder as builder
from recipe.time_series_forecast import reward, model_server
from recipe.time_series_forecast.train_policy_sft import encode_step, collate_responses
from recipe.time_series_forecast.trajectory_distillation import teacher_episode, select_trajectories


def task(horizon=3, dataset="ETTH1", length=12):
    timestamps = pd.date_range("2020-01-01", periods=length, freq="h").strftime("%Y-%m-%d %H:%M:%S").tolist()
    return ForecastTask(timestamps, [float(i) for i in range(length)], horizon,
                        dataset_name=dataset, target_name="OT",
                        historical_covariates={"load": [100. + i for i in range(length)]})


@pytest.mark.parametrize("order", [[0, 1, 2, 3, 4, 5], [2, 5, 4, 1, 3, 0]])
def test_grpo_broadcasts_terminal_advantage_after_batch_reordering(order):
    scores = torch.tensor([[0.], [0.], [.8], [0.], [0.], [.2]])
    trajectories = np.array(["a"] * 3 + ["b"] * 3)
    steps = np.array([0, 1, 2] * 2)
    mask = torch.tensor([[1, 1, 0]] * 6)
    rewards = torch.zeros_like(mask, dtype=torch.float)
    rewards[:, 1] = scores[:, 0]
    advantage, _ = compute_grpo_outcome_advantage(
        rewards[order], mask[order], np.array(["input"] * 6), trajectories[order],
        step_indices=steps[order], norm_adv_by_std_in_grpo=False,
    )
    for row, trajectory in enumerate(trajectories[order]):
        expected = .3 if trajectory == "a" else -.3
        assert advantage[row].tolist() == pytest.approx([expected, expected, 0.])
    normalized, _ = compute_grpo_outcome_advantage(
        rewards[order], mask[order], np.array(["input"] * 6), trajectories[order], step_indices=steps[order])
    assert normalized.abs().max().item() == pytest.approx(1 / np.sqrt(2), rel=1e-5)


def test_equal_outcomes_and_single_trajectory_have_zero_advantage():
    for uids in (np.array(["a", "b"]), np.array(["a", "a"])):
        advantage, _ = compute_grpo_outcome_advantage(torch.ones(2, 2), torch.ones(2, 2),
                                                    np.array(["q", "q"]), uids,
                                                    step_indices=np.array([0, 1]))
        assert torch.equal(advantage, torch.zeros_like(advantage))


def test_empty_final_response_still_assigns_failure_to_earlier_turns():
    mask = torch.tensor([[1., 1.], [0., 0.], [1., 1.], [1., 1.]])
    rewards = torch.tensor([[0., 0.], [-1., 0.], [0., 0.], [0., 1.]])
    advantages, _ = compute_grpo_outcome_advantage(
        rewards, mask, np.array(["q"] * 4), np.array(["a", "a", "b", "b"]),
        step_indices=np.array([0, 1, 0, 1]), norm_adv_by_std_in_grpo=False,
    )
    assert advantages.tolist() == [[-1., -1.], [0., 0.], [1., 1.], [1., 1.]]


def test_reward_matches_paper_weights_and_common_target_normalization():
    values = [1., 4., 2., 5., 1., 6., 2., 4., 1., 5., 3., 6.]
    dates = task().timestamps
    target = format_rows(dates, values)
    perfect = f"<answer>{target}</answer>"
    assert reward.compute_score("ETTH1", perfect, target) == pytest.approx(1.2)
    shifted = [v + 10 for v in values]
    mse_component = reward.compute_mse_score(format_rows(dates, shifted), target)
    assert mse_component == pytest.approx(.7 / (1 + np.log1p(100 / np.var(values))))
    assert reward.compute_score("ETTH1", target, target) == -1
    short = format_rows(dates[:6], values[:6])
    assert reward.compute_mse_score(short, target) == pytest.approx(.7 * .5)
    assert reward.compute_length_score(short, target) == pytest.approx(.1 * .5)
    assert np.isnan(reward.compute_raw_mse_mae(short, target)[0])


def test_failed_decomposition_only_zeroes_the_affected_reward_view(monkeypatch):
    target = format_rows(task().timestamps, list(map(float, range(12))))
    monkeypatch.setattr(reward, "mean_squared_error_season_trend", lambda *args: (0., None))
    assert reward.compute_season_trend_score(target, target) == pytest.approx(.05)


def test_large_finite_forecast_cannot_create_nan_group_rewards():
    dates = task().timestamps
    target = format_rows(dates, list(map(float, range(12))))
    with np.errstate(over="ignore", invalid="ignore"):
        score = reward.compute_score("ETTH1", f"<answer>{format_rows(dates, [1e300] * 12)}</answer>", target)
    assert np.isfinite(score) and 0. <= score <= 1.2


def test_state_contains_only_current_memory_and_recent_refinement_history():
    sample = task(length=20)
    sample.text_context = "Published before forecast origin"
    for analysis, prediction in (([], None), (["diagnosis"], None), (["diagnosis"], "forecast")):
        messages = build_state_messages(sample, analysis, prediction)
        assert [message["role"] for message in messages] == ["system", "user"]
        assert "Published before forecast origin" in messages[1]["content"]
        assert (sample.timestamps[0] in messages[1]["content"]) == (prediction is None)
        assert sample.timestamps[-1] in messages[1]["content"]


def test_task_schema_rejects_future_target_injection():
    payload = task().to_dict()
    payload["ground_truth"] = [999.] * 3
    with pytest.raises(ValueError, match="label"):
        ForecastTask.from_content(json.dumps(payload))
    sample = task().to_dict()
    sample["future_covariates"] = {"OT": [999.] * 3}
    with pytest.raises(ValueError, match="targets"):
        ForecastTask.from_content(json.dumps(sample))


@pytest.mark.parametrize("dataset,ratios", [("ETTH1", (6, 2, 2)), ("WIND", (7, 1, 2))])
def test_raw_chronological_boundaries_and_historical_covariates(tmp_path, dataset, ratios):
    n = 1000
    csv = tmp_path / "series.csv"
    pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n, freq="h"),
                  "OT": np.arange(n), "HUFL": np.arange(n) + 10000, "text": ["unused"] * n}).to_csv(csv, index=False)
    records = builder.build_time_series_records(csv, data_source=dataset, lookback_window=12,
                                               forecast_horizon=4, stride=4)
    splits = builder.split_records_contiguously(records)
    bounds = np.cumsum([n * x // 10 for x in ratios])
    start = 0
    for (name, frame), end in zip(splits.items(), bounds):
        for _, row in frame.iterrows():
            info = row["extra_info"]
            assert start <= info["target_start_index"] < info["target_end_index"] <= end
            sample = ForecastTask.from_content(row["prompt"][0]["content"])
            assert len(sample.historical_covariates["HUFL"]) == 12
            assert not sample.future_covariates
            assert not sample.text_context
        start = end


@pytest.mark.parametrize("entrypoint", [builder.build_and_save_banddrop_etth1_dataset,
                                      builder.build_and_save_curriculum_etth1_dataset])
def test_legacy_curriculum_entrypoints_never_score_or_filter_heldout_windows(tmp_path, entrypoint):
    csv = tmp_path / "ETTh1.csv"
    pd.DataFrame({"date": pd.date_range("2020-01-01", periods=2400, freq="h"),
                  "OT": np.arange(2400), "HUFL": np.arange(2400) + 10000}).to_csv(csv, index=False)
    raw = builder.split_records_contiguously(builder.build_time_series_records(csv))
    target_starts = []
    def reference(timestamps, values, target, horizon):
        target_starts.append(target[0])
        assert target[-1] < 1440
        return float(target[0])
    paths = entrypoint(csv_path=csv, output_dir=tmp_path / "output", teacher_mse_fn=reference)
    assert len(target_starts) == len(raw["train"])
    for split in ("val", "test"):
        saved = pd.read_parquet(paths[split])
        assert saved["reward_model"].map(lambda r: r["ground_truth"]).tolist() == raw[split]["reward_model"].map(lambda r: r["ground_truth"]).tolist()


def test_epf_keeps_only_forecast_fields_and_origin_text(tmp_path):
    csv = tmp_path / "epf.csv"
    pd.DataFrame({"date": pd.date_range("2020-01-01", periods=20, freq="h"),
                  "OT": np.arange(20) + 5000, "Grid load forecast": np.arange(20) + 100,
                  "measured_load": np.arange(20) + 200, "text": [f"text-{i}" for i in range(20)]}).to_csv(csv, index=False)
    records = builder.build_time_series_records(csv, data_source="NP", lookback_window=12, forecast_horizon=4)
    sample = ForecastTask.from_content(records.iloc[0]["prompt"][0]["content"])
    assert sample.future_covariates == {"Grid load forecast": [112., 113., 114., 115.]}
    assert sample.text_context == "text-11"
    assert "5012" not in sample.serialize()
    assert "text-12" not in sample.serialize()


def test_chronos_receives_covariates_without_future_target(monkeypatch):
    sample = task(dataset="NP")
    sample.future_covariates = {"load": [120., 121., 122.]}
    captured = {}
    class Pipeline:
        def predict_df(self, history, **kwargs):
            captured.update(history=history, **kwargs)
            return pd.DataFrame({"0.5": [1., 2., 3.]})
    monkeypatch.setitem(model_server._models, "chronos2", Pipeline())
    request = model_server.PredictRequest(**sample.model_request("chronos2"))
    result = model_server.predict_with_chronos2(request)
    assert result.timestamps == sample.future_timestamps
    assert "load" in captured["history"]
    assert "target" not in captured["future_df"]
    assert captured["future_df"]["load"].tolist() == [120., 121., 122.]


def test_neural_forecaster_receives_ordered_channels_and_returns_target(monkeypatch):
    sample = task()
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
        def forward(self, inputs):
            assert inputs.shape == (1, 12, 2)
            assert inputs[0, :, 1].tolist() == sample.historical_covariates["load"]
            return torch.tensor([[[1., 999.], [2., 999.], [3., 999.]]])
    monkeypatch.setattr(model_server, "load_itransformer", lambda *args, **kwargs: Model())
    monkeypatch.setitem(model_server._configs, "itransformer:ETTH1", {"enc_in": 2, "input_columns": ["OT", "load"]})
    result = model_server.predict_with_pytorch_model(model_server.PredictRequest(**sample.model_request("itransformer")), "itransformer")
    assert result.values == [1., 2., 3.]


class Tokenizer:
    eos_token_id = 99
    def apply_chat_template(self, messages, **kwargs):
        return [1, 2, 3]
    def encode(self, text, **kwargs):
        return [10, 11]


def test_sft_masks_states_and_padding_without_truncating_responses():
    step = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
            "response": "teacher response", "tools": []}
    row = encode_step(step, Tokenizer())
    assert row["labels"] == [-100, -100, -100, 10, 11, 99]
    padded = collate_responses([row, {k: v[:-1] for k, v in row.items()}], 0)
    assert padded["labels"][1, -1] == -100
    assert padded["attention_mask"][1, -1] == 0
    with pytest.raises(ValueError, match="budget"):
        encode_step(step, Tokenizer(), max_prompt_length=2)


def test_teacher_episode_uses_fresh_memory_states_and_complete_tools(monkeypatch):
    from recipe.time_series_forecast import forecast_harness
    sample = task()
    expected = format_rows(sample.future_timestamps, [1., 2., 3.])
    async def mock_predict(task, model):
        return expected
    monkeypatch.setattr(forecast_harness, "predict", mock_predict)
    messages_seen = []
    responses = iter(['<tool_call>{"name":"extract_basic_statistics","arguments":{}}</tool_call>',
                      '<tool_call>{"name":"predict_time_series","arguments":{"model_name":"chronos2"}}</tool_call>',
                      f'<think>refine</think><answer>{expected}</answer>'])
    async def generate(messages, tools):
        messages_seen.append(messages)
        return next(responses)
    episode = asyncio.run(teacher_episode(sample, generate))
    assert episode["complete"]
    assert episode["prediction"] == [1., 2., 3.]
    assert all([m["role"] for m in messages] == ["system", "user"] for messages in messages_seen)
    assert all(result["success"] for result in episode["tool_results"])


def test_sft_selection_is_balanced_by_dataset_and_raw_mse():
    def candidate(dataset, mse, split="train"):
        sample = task(dataset=dataset)
        forecast = format_rows(sample.future_timestamps, [1., 2., 3.])
        return {"task": sample.to_dict(), "trajectory_uid": f"{dataset}-{mse}-{split}",
                "data_source": dataset, "split": split, "mse": mse, "complete": True,
                "prediction": [1., 2., 3.],
                "tool_results": [{"name": "extract_basic_statistics", "success": True},
                                 {"name": "predict_time_series", "success": True, "observation": forecast}],
                "steps": [{"messages": build_state_messages(sample, [], None),
                           "response": '<tool_call>{"name":"extract_basic_statistics","arguments":{}}</tool_call>'},
                          {"messages": build_state_messages(sample, ["diagnostic"], None),
                           "response": '<tool_call>{"name":"predict_time_series","arguments":{"model_name":"chronos2"}}</tool_call>'},
                          {"messages": build_state_messages(sample, ["diagnostic"], forecast),
                           "response": f"<think>refine</think><answer>{forecast}</answer>"}]}
    candidates = [candidate(d, mse) for d in ("ETTH1", "NP") for mse in (8., 1., 3.)]
    candidates.append(candidate("NP", 0., split="test"))
    invalid = candidate("NP", 0.)
    invalid["steps"][-1]["response"] = "<think>bad</think><answer>nan</answer>"
    candidates.extend([invalid, candidates[0]])  # Malformed output and repeated UID.
    selected = select_trajectories(candidates, datasets=("ETTH1", "NP"), per_dataset=2)
    assert [(x["data_source"], x["mse"]) for x in selected] == [("ETTH1", 1.), ("ETTH1", 3.), ("NP", 1.), ("NP", 3.)]
    with pytest.raises(ValueError, match="need"):
        select_trajectories(candidates, datasets=("ETTH1", "NP"), per_dataset=4)


def test_paper_flow_requires_policy_tool_choices_and_one_forecasting_call():
    from recipe.time_series_forecast.time_series_forecast_agent_flow import TimeSeriesForecastAgentFlow
    flow = TimeSeriesForecastAgentFlow.__new__(TimeSeriesForecastAgentFlow)
    flow.allow_tool_fallback = False
    flow.basic_statistics = {"mean": 1.}
    flow.within_channel_dynamics = flow.forecast_residuals = flow.data_quality = flow.event_summary = None
    flow.prediction_results = None
    from recipe.time_series_forecast.prompts import TIMESERIES_TOOL_SCHEMAS
    flow.tool_schemas = TIMESERIES_TOOL_SCHEMAS
    assert not flow._should_auto_predict_after_tool_markup_failure("<tool_call>bad", [])
    with pytest.raises(ValueError, match="explicitly select"):
        flow._normalize_predict_model_name("unknown")
    calls, skipped = flow._filter_tool_calls_for_current_stage(
        [SimpleNamespace(name="predict_time_series"), SimpleNamespace(name="predict_time_series")])
    assert calls == [] and skipped
