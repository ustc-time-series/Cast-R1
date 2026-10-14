"""Generate training-only teacher episodes and select 120 complete episodes per dataset."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from uuid import uuid4

import httpx
import numpy as np
import pandas as pd

from recipe.time_series_forecast.forecast_task import ForecastTask
from recipe.time_series_forecast.forecast_harness import DIAGNOSTICS, ForecastHarness, parse_tool_calls, validate_forecast
from recipe.time_series_forecast.reward import extract_answer


DATASETS = ("ETTH1", "ETTH2", "ETTM1", "ETTM2", "WIND", "NP", "PJM", "BE", "DE", "FR")


async def teacher_episode(task: ForecastTask, generate, *, max_steps: int = 5) -> dict:
    harness = ForecastHarness(task)
    steps = []
    for _ in range(max_steps):
        messages, tools = harness.messages(), harness.tools()
        response = await generate(messages, tools)
        steps.append({"messages": messages, "tools": tools, "response": response})
        calls = parse_tool_calls(response)
        if "<answer>" in response and "</answer>" in response:
            if (calls or not harness.analysis or harness.prediction is None
                    or not re.fullmatch(r"\s*<think>.*?</think>\s*<answer>.*?</answer>\s*", response, re.DOTALL)):
                raise ValueError("Teacher did not complete diagnosis, model selection and refinement")
            values = validate_forecast(task, extract_answer(response))
            return {"task": task.to_dict(), "steps": steps, "prediction": values,
                    "tool_results": harness.tool_results, "complete": True}
        await harness.execute(calls)
    raise ValueError("Teacher episode exhausted its budget without a final forecast")


def validate_complete_trajectory(candidate: dict) -> None:
    """Recheck execution and final output instead of trusting a cached success flag."""
    if candidate.get("split") != "train" or not candidate.get("complete"):
        raise ValueError("Only complete training trajectories can be distilled")
    task = ForecastTask.from_content(json.dumps(candidate["task"], allow_nan=False))
    if task.dataset_name.upper() != candidate["data_source"].upper():
        raise ValueError("Trajectory dataset does not match its task")
    steps, tools = candidate["steps"], candidate["tool_results"]
    if not 3 <= len(steps) <= 5 or not tools or not all(tool.get("success") for tool in tools):
        raise ValueError("Incomplete or failed trajectory")
    if any([message["role"] for message in step["messages"]] != ["system", "user"] for step in steps):
        raise ValueError("Trajectory states must contain fresh system/user messages")
    calls = []
    for step in steps[:-1]:
        step_calls = parse_tool_calls(step["response"])
        if not step_calls or len(step_calls) > 5 or "<answer>" in step["response"]:
            raise ValueError("Malformed nonterminal tool response")
        calls.extend(step_calls)
    if ([call["name"] for call in calls] != [tool["name"] for tool in tools]
            or tools[-1]["name"] != "predict_time_series"
            or any(tool["name"] not in DIAGNOSTICS for tool in tools[:-1])):
        raise ValueError("Recorded tool execution does not match the teacher actions")
    if calls[-1].get("arguments", {}).get("model_name") not in {"arima", "chronos2", "patchtst", "itransformer"}:
        raise ValueError("Malformed forecasting model selection")
    model_calls = [tool for tool in tools if tool["name"] == "predict_time_series"]
    if len(model_calls) != 1 or not any(tool["name"] in DIAGNOSTICS for tool in tools):
        raise ValueError("Trajectory must complete diagnosis and one model selection")
    validate_forecast(task, model_calls[0]["observation"])
    final = steps[-1]["response"]
    if (not re.fullmatch(r"\s*<think>.*?</think>\s*<answer>.*?</answer>\s*", final, re.DOTALL)
            or parse_tool_calls(final)):
        raise ValueError("Malformed final refinement response")
    values = validate_forecast(task, extract_answer(final))
    if values != candidate["prediction"] or not np.isfinite(candidate["mse"]) or candidate["mse"] < 0:
        raise ValueError("Invalid recorded prediction or forecasting error")


def select_trajectories(candidates: list[dict], *, datasets=DATASETS, per_dataset: int = 120,
                        validate_step=None) -> list[dict]:
    selected = []
    for dataset in datasets:
        valid, seen = [], set()
        for candidate in candidates:
            if candidate.get("data_source", "").upper() != dataset or candidate.get("split") != "train":
                continue
            try:
                validate_complete_trajectory(candidate)
                if validate_step is not None:
                    for step in candidate["steps"]:
                        validate_step(step)
            except (ValueError, KeyError, TypeError, AttributeError):
                continue
            uid = candidate.get("trajectory_uid")
            if not uid or uid in seen:
                continue
            seen.add(uid)
            valid.append(candidate)
        valid.sort(key=lambda item: (item["mse"], item.get("trajectory_uid", "")))
        if len(valid) < per_dataset:
            raise ValueError(f"{dataset}: {len(valid)} valid complete trajectories; need {per_dataset}")
        selected.extend(valid[:per_dataset])
    return selected


async def generate_candidates(args) -> None:
    data = pd.read_parquet(args.inputs)
    if any(info.get("split") != "train" for info in data["extra_info"]):
        raise ValueError("Teacher trajectories must be generated exclusively from training-split windows")
    api_key = os.environ.get(args.api_key_env)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with httpx.AsyncClient(timeout=300, headers=headers) as client:
        async def generate(messages, tools):
            payload = {"model": args.teacher_model, "messages": messages, "temperature": 1.0,
                       "max_tokens": 2048}
            if tools:
                payload["tools"] = tools
            result = await client.post(f"{args.api_base.rstrip('/')}/chat/completions", json=payload)
            result.raise_for_status()
            message = result.json()["choices"][0]["message"]
            text = message.get("content") or ""
            for call in message.get("tool_calls") or []:
                function = call["function"]
                arguments = function["arguments"]
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                text += "\n<tool_call>" + json.dumps({"name": function["name"], "arguments": arguments}) + "</tool_call>"
            return text

        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Append lets an interrupted expensive generation resume without losing candidates.
        with args.output.open("a", encoding="utf-8") as handle:
            for _, row in data.iterrows():
                task = ForecastTask.from_content(row["prompt"][0]["content"])
                target_task = ForecastTask.from_content(row["reward_model"]["ground_truth"], horizon=task.horizon)
                if len(target_task.values) != task.horizon or target_task.timestamps != task.future_timestamps:
                    raise ValueError("Training target does not match the forecasting horizon")
                for _ in range(args.rollouts_per_window):
                    try:
                        candidate = await teacher_episode(task, generate, max_steps=args.max_steps)
                    except httpx.HTTPStatusError as exc:
                        if exc.response.status_code in {401, 403, 404}:
                            raise
                        print(f"Discarded {task.dataset_name} episode: HTTP {exc.response.status_code}", flush=True)
                        continue
                    except (ValueError, httpx.HTTPError, KeyError, json.JSONDecodeError) as exc:
                        print(f"Discarded {task.dataset_name} episode: {type(exc).__name__}", flush=True)
                        continue
                    candidate.update({"trajectory_uid": uuid4().hex, "data_source": task.dataset_name.upper(),
                                      "split": "train", "teacher": args.teacher_model,
                                      "mse": float(np.mean((np.asarray(candidate["prediction"]) - target_task.values) ** 2))})
                    handle.write(json.dumps(candidate, ensure_ascii=False, allow_nan=False) + "\n")
                    handle.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--inputs", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--api-base", required=True, help="OpenAI-compatible teacher endpoint, ending in /v1")
    generate.add_argument("--api-key-env", default="TEACHER_API_KEY")
    generate.add_argument("--teacher-model", default="DeepSeek-V3.2")
    generate.add_argument("--rollouts-per-window", type=int, default=4)
    generate.add_argument("--max-steps", type=int, default=5)
    select = commands.add_parser("select")
    select.add_argument("--inputs", type=Path, nargs="+", required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--student-tokenizer", required=True)
    select.add_argument("--per-dataset", type=int, default=120)
    args = parser.parse_args()
    if args.command == "generate":
        asyncio.run(generate_candidates(args))
    else:
        from transformers import AutoTokenizer
        from recipe.time_series_forecast.train_policy_sft import encode_step
        tokenizer = AutoTokenizer.from_pretrained(args.student_tokenizer)
        candidates = [json.loads(line) for path in args.inputs for line in path.read_text().splitlines() if line.strip()]
        selected = select_trajectories(candidates, per_dataset=args.per_dataset,
                                      validate_step=lambda step: encode_step(step, tokenizer))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            for episode in selected:
                handle.write(json.dumps(episode, ensure_ascii=False, allow_nan=False) + "\n")
        print(f"Selected {len(selected)} complete trajectories / {sum(len(t['steps']) for t in selected)} responses")


if __name__ == "__main__":
    main()
