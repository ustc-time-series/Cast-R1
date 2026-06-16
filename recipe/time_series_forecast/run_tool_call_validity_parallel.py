#!/usr/bin/env python3
"""
Launch tool-call validity benchmarks in parallel across multiple GPUs and merge
the resulting shard outputs into one aggregate report.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recipe.time_series_forecast.benchmark_tool_call_validity import (  # noqa: E402
    aggregate_results,
)


DEFAULT_SHARDS = [
    {"card": 0, "dataset": "etth1", "offset": 0, "tag": "etth1_000_019"},
    {"card": 1, "dataset": "etth1", "offset": 20, "tag": "etth1_020_039"},
    {"card": 2, "dataset": "etth1", "offset": 40, "tag": "etth1_040_059"},
    {"card": 3, "dataset": "np", "offset": 0, "tag": "np_000_019"},
    {"card": 4, "dataset": "np", "offset": 20, "tag": "np_020_039"},
    {"card": 5, "dataset": "np", "offset": 40, "tag": "np_040_059"},
    {"card": 6, "dataset": "wind", "offset": 0, "tag": "wind_000_019"},
    {"card": 7, "dataset": "wind", "offset": 20, "tag": "wind_020_039"},
]


def format_shards(shards: list[dict[str, Any]]) -> str:
    return ",".join(
        f"{item['card']}:{item['dataset']}:{item['offset']}:{item['tag']}" for item in shards
    )


def parse_shards(text: str) -> list[dict[str, Any]]:
    shards = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split(":")
        if len(parts) != 4:
            raise ValueError(f"Invalid shard spec: {raw}")
        card, dataset, offset, tag = parts
        shards.append(
            {
                "card": int(card),
                "dataset": dataset,
                "offset": int(offset),
                "tag": tag,
            }
        )
    if not shards:
        raise ValueError("No shards specified")
    return shards


def run_parallel(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    benchmark_script = PROJECT_ROOT / "recipe" / "time_series_forecast" / "benchmark_tool_call_validity.py"
    python_exe = sys.executable

    procs: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    for shard in args.shards:
        tag = shard["tag"]
        json_path = output_dir / f"{tag}.json"
        md_path = output_dir / f"{tag}.md"
        log_path = log_dir / f"{tag}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(shard["card"])
        env["PYTHONUNBUFFERED"] = "1"
        cmd = [
            python_exe,
            str(benchmark_script),
            "--datasets",
            shard["dataset"],
            "--num-samples",
            str(args.num_samples),
            "--num-warmup",
            str(args.num_warmup),
            "--sample-offset",
            str(shard["offset"]),
            "--output-json",
            str(json_path),
            "--output-md",
            str(md_path),
            "--model",
            args.model,
            "--tp",
            str(args.tp),
            "--max-steps",
            str(args.max_steps),
            "--max-model-len",
            str(args.max_model_len),
            "--max-tokens",
            str(args.max_tokens),
            "--gpu-mem",
            str(args.gpu_mem),
            "--temperature",
            str(args.temperature),
        ]
        log_f = open(log_path, "w")
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
        procs.append(
            {
                "proc": proc,
                "log_file": log_f,
                "log_path": str(log_path),
                "json_path": str(json_path),
                "md_path": str(md_path),
                "shard": shard,
                "cmd": cmd,
            }
        )
        print(f"[launch] card={shard['card']} dataset={shard['dataset']} offset={shard['offset']} pid={proc.pid}")

    failures = []
    shard_outputs = []
    try:
        while procs:
            still_running = []
            for item in procs:
                ret = item["proc"].poll()
                if ret is None:
                    still_running.append(item)
                    continue
                item["log_file"].close()
                shard = item["shard"]
                elapsed = time.perf_counter() - start_time
                print(
                    f"[done] card={shard['card']} dataset={shard['dataset']} offset={shard['offset']} "
                    f"exit={ret} elapsed={elapsed:.1f}s"
                )
                if ret != 0:
                    failures.append(
                        {
                            "shard": shard,
                            "exit_code": ret,
                            "log_path": item["log_path"],
                            "cmd": item["cmd"],
                        }
                    )
                else:
                    shard_outputs.append(item)
            procs = still_running
            if procs:
                time.sleep(10)
    finally:
        for item in procs:
            if item["proc"].poll() is None:
                item["proc"].terminate()
            item["log_file"].close()

    if failures:
        return {
            "status": "failed",
            "failures": failures,
            "elapsed_s": time.perf_counter() - start_time,
        }

    shard_jsons = []
    all_episodes = []
    for item in sorted(shard_outputs, key=lambda x: x["shard"]["tag"]):
        with open(item["json_path"], "r") as f:
            data = json.load(f)
        shard_jsons.append(
            {
                "shard": item["shard"],
                "json_path": item["json_path"],
                "md_path": item["md_path"],
                "log_path": item["log_path"],
                "summary": data["summary"],
            }
        )
        all_episodes.extend(data["episodes"])

    metrics = aggregate_results(all_episodes)
    elapsed = time.perf_counter() - start_time
    aggregate_json = {
        "config": {
            "model": args.model,
            "num_samples_per_shard": args.num_samples,
            "num_warmup": args.num_warmup,
            "max_steps": args.max_steps,
            "tp": args.tp,
            "temperature": args.temperature,
            "max_model_len": args.max_model_len,
            "max_tokens": args.max_tokens,
            "gpu_mem": args.gpu_mem,
            "shards": args.shards,
            "metric_denominator": "total_emitted_tool_call_attempts",
            "runner": str(Path(__file__).resolve()),
            "parallel_cards": [item["card"] for item in args.shards],
        },
        "elapsed_s": elapsed,
        "status": "completed",
        "summary": {
            "overall": metrics["overall"],
            "by_dataset": metrics["by_dataset"],
            "detail_breakdown": metrics["detail_breakdown"],
        },
        "shards": shard_jsons,
        "episodes": all_episodes,
    }
    return aggregate_json


def build_parallel_markdown(result: dict[str, Any]) -> str:
    overall = result["summary"]["overall"]
    lines = [
        "# Cast-R1 Tool-Call Validity Benchmark",
        "",
        f"- Model: `{result['config']['model']}`",
        "- Execution: parallel shard runs merged after completion",
        f"- Parallel cards: `{','.join(str(x) for x in result['config']['parallel_cards'])}`",
        f"- Shards: `{len(result['config']['shards'])}`",
        f"- Samples per shard: `{result['config']['num_samples_per_shard']}`",
        f"- Total episodes: `{overall['episodes']}`",
        f"- Elapsed time (s): `{result['elapsed_s']:.2f}`",
        "- Rate denominator: total emitted tool-call attempts",
        "- Redundant calls are a subset of syntactically valid calls and therefore overlap with valid-call counts",
        "",
        "## Overall",
        "",
        f"- Episodes: `{overall['episodes']}`",
        f"- Tool-call attempts: `{overall['tool_call_attempts']}`",
        f"- Valid calls: `{overall['valid_calls']}` ({overall['valid_call_rate']:.6f})",
        f"- Illegal call rate: `{overall['illegal_call_rate']:.6f}` ({overall['illegal_calls']}/{overall['tool_call_attempts']})",
        f"- Format error rate: `{overall['format_error_rate']:.6f}` ({overall['format_errors']}/{overall['tool_call_attempts']})",
        f"- Empty call rate: `{overall['empty_call_rate']:.6f}` ({overall['empty_calls']}/{overall['tool_call_attempts']})",
        f"- Duplicate / redundant call rate: `{overall['redundant_call_rate']:.6f}` ({overall['redundant_calls']}/{overall['tool_call_attempts']})",
        f"- Answer success rate: `{overall['answer_success_rate']:.6f}` ({overall['episodes_with_answer']}/{overall['episodes']})",
        "",
        "## By Dataset",
        "",
        "| Dataset | Episodes | Attempts | Illegal | Format | Empty | Redundant | Valid |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset, row in result["summary"]["by_dataset"].items():
        lines.append(
            f"| {dataset} | {row['episodes']} | {row['tool_call_attempts']} | "
            f"{row['illegal_call_rate']:.6f} | {row['format_error_rate']:.6f} | "
            f"{row['empty_call_rate']:.6f} | {row['redundant_call_rate']:.6f} | "
            f"{row['valid_call_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Shards",
            "",
            "| Tag | Card | Dataset | Offset | JSON | Log |",
            "| --- | ---: | --- | ---: | --- | --- |",
        ]
    )
    for shard in result["shards"]:
        meta = shard["shard"]
        lines.append(
            f"| {meta['tag']} | {meta['card']} | {meta['dataset']} | {meta['offset']} | "
            f"`{shard['json_path']}` | `{shard['log_path']}` |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(PROJECT_ROOT / "models" / "Qwen3-8B"))
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--num-warmup", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=3)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument(
        "--shards",
        default=format_shards(DEFAULT_SHARDS),
        help="comma-separated list of card:dataset:offset:tag specs",
    )
    ap.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "benchmark_results" / "tool_call_validity_parallel"),
    )
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", required=True)
    args = ap.parse_args()

    args.shards = parse_shards(args.shards)
    result = run_parallel(args)

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    if result["status"] == "completed":
        output_md.write_text(build_parallel_markdown(result))
        print(json.dumps(result["summary"]["overall"], indent=2, ensure_ascii=False))
        print(f"Wrote {output_json}")
        print(f"Wrote {output_md}")
        return

    output_md.write_text(
        "# Parallel Tool-Call Validity Benchmark\n\n"
        "Run failed.\n\n"
        "```json\n"
        f"{json.dumps(result, indent=2, ensure_ascii=False)}\n"
        "```\n"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
