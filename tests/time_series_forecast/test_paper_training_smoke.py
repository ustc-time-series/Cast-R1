"""CPU integration checks with tiny local weights; no network or paper metrics."""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from recipe.time_series_forecast.forecast_task import ForecastTask, format_rows


def sample_task():
    dates = pd.date_range("2020-01-01", periods=16, freq="h").strftime("%Y-%m-%d %H:%M:%S").tolist()
    return ForecastTask(dates, list(map(float, range(16))), 4, dataset_name="ETTH1",
                        target_name="OT", historical_covariates={"load": list(map(float, range(100, 116)))})


def test_multivariate_neural_training_checkpoint_and_service(tmp_path, monkeypatch):
    from recipe.time_series_forecast.train_small_models_from_rl import SmallModelTrainingConfig, train_and_save
    from recipe.time_series_forecast import model_server
    torch.set_num_threads(1)
    task = sample_task()
    row = {"prompt": [{"role": "user", "content": task.serialize()}],
           "reward_model": {"ground_truth": format_rows(task.future_timestamps, [16., 17., 18., 19.])}}
    data = tmp_path / "windows.parquet"
    pd.DataFrame([row, row]).to_parquet(data, index=False)
    monkeypatch.setattr(model_server, "_ACTIVE_DEVICE", "cpu")
    monkeypatch.setattr(model_server, "_models", {})
    monkeypatch.setattr(model_server, "_configs", {})
    for model in ("itransformer", "patchtst"):
        root = tmp_path / model
        monkeypatch.setitem(model_server._MODEL_DIRS, model, root)
        metrics = train_and_save(SmallModelTrainingConfig(
            model, data, data, root / "ETTH1" / "checkpoint.pth", epochs=1, batch_size=2,
            device="cpu", seq_len=16, pred_len=4, d_model=8, n_heads=2, e_layers=1,
            d_ff=16, patch_len=4, stride=2,
        ))
        assert np.isfinite(metrics["train_loss"]) and np.isfinite(metrics["val_loss"])
        config = json.loads((root / "ETTH1" / "config.json").read_text())
        assert config["enc_in"] == 2 and config["input_columns"] == ["OT", "load"]
        result = model_server.predict_with_pytorch_model(model_server.PredictRequest(**task.model_request(model)), model)
        assert result.timestamps == task.future_timestamps
        assert len(result.values) == 4 and np.isfinite(result.values).all()


def test_response_only_sft_entrypoint_trains_and_saves_local_tiny_policy(tmp_path, monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from recipe.time_series_forecast import forecast_harness
    from recipe.time_series_forecast.trajectory_distillation import teacher_episode
    task = sample_task()
    forecast = format_rows(task.future_timestamps, [16., 17., 18., 19.])
    async def fake_predict(task, model):
        return forecast
    monkeypatch.setattr(forecast_harness, "predict", fake_predict)
    responses = iter(['<tool_call>{"name":"extract_basic_statistics","arguments":{}}</tool_call>',
                      '<tool_call>{"name":"predict_time_series","arguments":{"model_name":"chronos2"}}</tool_call>',
                      f'<think>retain forecast</think><answer>{forecast}</answer>'])
    async def generate(messages, tools):
        return next(responses)
    episode = asyncio.run(teacher_episode(task, generate))
    episode.update(split="train", data_source="ETTH1", mse=0., trajectory_uid="smoke")
    corpus = tmp_path / "episodes.jsonl"
    corpus.write_text(json.dumps(episode) + "\n")
    vocabulary = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, "system": 3, "user": 4,
                  "assistant": 5, "think": 6, "answer": 7, "tool_call": 8}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]",
                                        unk_token="[UNK]", eos_token="[EOS]")
    tokenizer.chat_template = "{% for m in messages %}{{m['role'] + ' ' + m['content'] + '\\n'}}{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    base = tmp_path / "tiny-policy"
    tokenizer.save_pretrained(base)
    GPT2LMHeadModel(GPT2Config(vocab_size=len(vocabulary), n_positions=2048, n_embd=16,
                             n_layer=1, n_head=2, eos_token_id=2, pad_token_id=0)).save_pretrained(base)
    output = tmp_path / "sft"
    command = [sys.executable, "-m", "recipe.time_series_forecast.train_policy_sft", "--cpu",
               "--model", str(base), "--trajectories", str(corpus), "--output", str(output),
               "--global-batch-size", "2"]
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[2],
                            env={**os.environ, "OMP_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"},
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((output / "training_manifest.json").read_text())
    assert manifest["num_trajectories"] == 1 and manifest["response_examples"] == 3
    assert manifest["precision"] == "float32" and manifest["dataset_counts"] == {"ETTH1": 1}
    assert (output / "model.safetensors").is_file()
