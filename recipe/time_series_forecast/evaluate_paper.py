"""Evaluate full-horizon raw-scale MSE/MAE using the shared memory-only harness."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from recipe.time_series_forecast.forecast_task import ForecastTask, parse_rows
from recipe.time_series_forecast.trajectory_distillation import teacher_episode


async def evaluate(args):
    data = pd.read_parquet(args.inputs)
    if any(info.get("split") != "test" for info in data["extra_info"]):
        raise ValueError("Final evaluation requires untouched test-split windows")
    key = os.environ.get(args.api_key_env)
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    results = []
    async with httpx.AsyncClient(headers=headers, timeout=300) as client:
        async def generate(messages, tools):
            payload = {"model": args.model, "messages": messages, "temperature": 0.0, "max_tokens": 4096}
            if tools:
                payload["tools"] = tools
            response = await client.post(f"{args.api_base.rstrip('/')}/chat/completions", json=payload)
            response.raise_for_status()
            message = response.json()["choices"][0]["message"]
            text = message.get("content") or ""
            for call in message.get("tool_calls") or []:
                arguments = call["function"]["arguments"]
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                text += "\n<tool_call>" + json.dumps({"name": call["function"]["name"], "arguments": arguments}) + "</tool_call>"
            return text
        for _, row in data.iterrows():
            task = ForecastTask.from_content(row["prompt"][0]["content"])
            # Targets are accessed only for metrics, never supplied to policy/tools.
            target_times, target = parse_rows(row["reward_model"]["ground_truth"])
            if target_times != task.future_timestamps:
                raise ValueError("Test target horizon/grid does not match input task")
            record = {"dataset": task.dataset_name, "complete": False}
            try:
                episode = await teacher_episode(task, generate, max_steps=args.max_steps)
                difference = np.asarray(episode["prediction"]) - target
                record.update(complete=True, mse=float(np.mean(difference ** 2)),
                              mae=float(np.mean(np.abs(difference))), prediction=episode["prediction"])
            except (ValueError, KeyError, httpx.HTTPError) as exc:
                record["failure"] = type(exc).__name__
            results.append(record)
    summary = {}
    for dataset in sorted({r["dataset"] for r in results}):
        episodes = [r for r in results if r["dataset"] == dataset]
        complete = [r for r in episodes if r["complete"]]
        summary[dataset] = {"episodes": len(episodes), "completed": len(complete),
                            "failure_rate": 1 - len(complete) / len(episodes),
                            "mse": float(np.mean([r["mse"] for r in complete])) if len(complete) == len(episodes) else None,
                            "mae": float(np.mean([r["mae"] for r in complete])) if len(complete) == len(episodes) else None,
                            "mse_on_complete_horizons": float(np.mean([r["mse"] for r in complete])) if complete else None,
                            "mae_on_complete_horizons": float(np.mean([r["mae"] for r in complete])) if complete else None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "episodes": results}, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--api-base", required=True, help="OpenAI-compatible serving endpoint for the trained student")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="STUDENT_API_KEY")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=5)
    asyncio.run(evaluate(parser.parse_args()))


if __name__ == "__main__":
    main()
