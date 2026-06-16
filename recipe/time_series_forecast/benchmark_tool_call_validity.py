#!/usr/bin/env python3
"""
Tool-call validity benchmark for the Cast-R1 time-series agent.

This script replays the same 3-turn tool-use protocol as benchmark_castr1.py,
but preserves raw assistant outputs and computes call-level robustness stats:

- illegal call rate
- format error rate
- empty call rate
- duplicate / redundant call rate

By default it evaluates the Qwen3-8B Cast-R1 agent on the three datasets used
in the rebuttal latency benchmark: ETTH1, NP, and Wind.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch_npu  # noqa: F401
except Exception:
    pass

from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

from recipe.time_series_forecast.benchmark_castr1 import (  # noqa: E402
    EXTRACT_TOOL_FNS,
    build_user_prompt,
    extract_answer,
    load_samples,
)
from recipe.time_series_forecast.prompts import (  # noqa: E402
    TIMESERIES_SYSTEM_PROMPT,
    TIMESERIES_TOOL_SCHEMAS,
)
from recipe.time_series_forecast.utils import (  # noqa: E402
    parse_time_series_string,
)


DATASET_PRESETS = {
    "etth1": {
        "display": "ETTH1",
        "path": str(PROJECT_ROOT / "datasets" / "ETTH1" / "test.parquet"),
        "lookback": 96,
        "horizon": 96,
    },
    "np": {
        "display": "NP",
        "path": str(PROJECT_ROOT / "datasets" / "NP" / "test.parquet"),
        "lookback": 96,
        "horizon": 96,
    },
    "wind": {
        "display": "Wind",
        "path": str(PROJECT_ROOT / "datasets" / "Wind" / "test.parquet"),
        "lookback": 96,
        "horizon": 96,
    },
}

TOOL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
TOOL_OPEN_RE = re.compile(r"<tool_call>", re.IGNORECASE)
TOOL_CLOSE_RE = re.compile(r"</tool_call>", re.IGNORECASE)

FEATURE_TOOLS = {
    "extract_basic_statistics",
    "extract_within_channel_dynamics",
    "extract_forecast_residuals",
    "extract_data_quality",
    "extract_event_summary",
}
ALLOWED_TOOLS = FEATURE_TOOLS | {"predict_time_series"}
ALLOWED_PREDICT_MODELS = {"chronos2", "patchtst", "itransformer", "arima"}


@dataclass
class CallRecord:
    dataset: str
    sample_index: int
    turn_index: int
    raw_payload: str
    call_kind: str
    name: str | None = None
    arguments: Any = None
    detail: str | None = None
    redundant: bool = False


def format_float(value: float) -> str:
    return f"{value:.6f}"


def make_stub_prediction(timestamps: list[Any], values: list[float], horizon: int) -> str:
    if not timestamps or not values:
        return ""

    try:
        import pandas as pd

        parsed_timestamps = [pd.to_datetime(ts) for ts in timestamps]
    except Exception:
        parsed_timestamps = timestamps

    ts0 = parsed_timestamps[-2] if len(parsed_timestamps) >= 2 else parsed_timestamps[-1]
    ts1 = parsed_timestamps[-1]
    try:
        delta = ts1 - ts0
    except Exception:
        from datetime import timedelta

        delta = timedelta(hours=1)

    if getattr(delta, "total_seconds", None):
        try:
            if delta.total_seconds() <= 0:
                raise ValueError("non-positive delta")
        except Exception:
            from datetime import timedelta

            delta = timedelta(hours=1)

    last_val = float(values[-1])
    out_lines = []
    for i in range(horizon):
        ts = parsed_timestamps[-1] + delta * (i + 1)
        if hasattr(ts, "strftime"):
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
        else:
            ts_str = str(ts)
        out_lines.append(f"{ts_str} {last_val:.3f}")
    return "\n".join(out_lines)


def execute_local_tool(name: str, arguments: dict[str, Any], ctx: dict[str, Any]) -> str:
    if name in EXTRACT_TOOL_FNS:
        extract_fn, format_fn = EXTRACT_TOOL_FNS[name]
        feats = extract_fn(data=ctx["values"])
        ctx["features_done"] = True
        return format_fn(feats)
    if name == "predict_time_series":
        if not ctx["features_done"]:
            return "ERROR: please call a feature extraction tool before predict_time_series"
        return make_stub_prediction(ctx["timestamps"], ctx["values"], ctx["horizon"])
    return f"ERROR: unknown tool {name}"


def normalize_signature(name: str, arguments: dict[str, Any]) -> str:
    return json.dumps(
        {"name": name, "arguments": arguments},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def classify_tool_payload(
    *,
    payload: str,
    dataset: str,
    sample_index: int,
    turn_index: int,
    features_done: bool,
    seen_signatures: set[str],
) -> CallRecord:
    stripped = payload.strip()
    if not stripped:
        return CallRecord(
            dataset=dataset,
            sample_index=sample_index,
            turn_index=turn_index,
            raw_payload=payload,
            call_kind="empty",
            detail="empty_block",
        )

    try:
        obj = json.loads(stripped)
    except Exception:
        return CallRecord(
            dataset=dataset,
            sample_index=sample_index,
            turn_index=turn_index,
            raw_payload=payload,
            call_kind="format_error",
            detail="json_decode_error",
        )

    if not isinstance(obj, dict):
        return CallRecord(
            dataset=dataset,
            sample_index=sample_index,
            turn_index=turn_index,
            raw_payload=payload,
            call_kind="format_error",
            detail="json_not_object",
        )

    name = obj.get("name")
    arguments = obj.get("arguments", {})

    if isinstance(arguments, str):
        if not arguments.strip():
            arguments = {}
        else:
            try:
                arguments = json.loads(arguments)
            except Exception:
                return CallRecord(
                    dataset=dataset,
                    sample_index=sample_index,
                    turn_index=turn_index,
                    raw_payload=payload,
                    call_kind="format_error",
                    name=str(name) if name is not None else None,
                    detail="arguments_json_decode_error",
                )

    if name is None or not str(name).strip():
        return CallRecord(
            dataset=dataset,
            sample_index=sample_index,
            turn_index=turn_index,
            raw_payload=payload,
            call_kind="empty",
            detail="empty_name",
        )

    name = str(name).strip()

    if not isinstance(arguments, dict):
        return CallRecord(
            dataset=dataset,
            sample_index=sample_index,
            turn_index=turn_index,
            raw_payload=payload,
            call_kind="illegal",
            name=name,
            arguments=arguments,
            detail="arguments_not_object",
        )

    if name not in ALLOWED_TOOLS:
        return CallRecord(
            dataset=dataset,
            sample_index=sample_index,
            turn_index=turn_index,
            raw_payload=payload,
            call_kind="illegal",
            name=name,
            arguments=arguments,
            detail="unknown_tool",
        )

    if name in FEATURE_TOOLS:
        if arguments:
            return CallRecord(
                dataset=dataset,
                sample_index=sample_index,
                turn_index=turn_index,
                raw_payload=payload,
                call_kind="illegal",
                name=name,
                arguments=arguments,
                detail="unexpected_feature_arguments",
            )
    elif name == "predict_time_series":
        model_name = arguments.get("model_name")
        if model_name is None or (isinstance(model_name, str) and not model_name.strip()):
            return CallRecord(
                dataset=dataset,
                sample_index=sample_index,
                turn_index=turn_index,
                raw_payload=payload,
                call_kind="empty",
                name=name,
                arguments=arguments,
                detail="missing_model_name",
            )
        if not isinstance(model_name, str):
            return CallRecord(
                dataset=dataset,
                sample_index=sample_index,
                turn_index=turn_index,
                raw_payload=payload,
                call_kind="illegal",
                name=name,
                arguments=arguments,
                detail="non_string_model_name",
            )
        model_name = model_name.strip()
        arguments = {"model_name": model_name}
        if model_name not in ALLOWED_PREDICT_MODELS:
            return CallRecord(
                dataset=dataset,
                sample_index=sample_index,
                turn_index=turn_index,
                raw_payload=payload,
                call_kind="illegal",
                name=name,
                arguments=arguments,
                detail="invalid_model_name",
            )
        if not features_done:
            return CallRecord(
                dataset=dataset,
                sample_index=sample_index,
                turn_index=turn_index,
                raw_payload=payload,
                call_kind="illegal",
                name=name,
                arguments=arguments,
                detail="predict_before_features",
            )

    signature = normalize_signature(name, arguments)
    redundant = signature in seen_signatures
    return CallRecord(
        dataset=dataset,
        sample_index=sample_index,
        turn_index=turn_index,
        raw_payload=payload,
        call_kind="valid",
        name=name,
        arguments=arguments,
        detail="ok",
        redundant=redundant,
    )


def analyze_response_text(
    *,
    text: str,
    dataset: str,
    sample_index: int,
    turn_index: int,
    features_done: bool,
    seen_signatures: set[str],
) -> list[CallRecord]:
    records: list[CallRecord] = []
    blocks = list(TOOL_BLOCK_RE.finditer(text))
    open_count = len(TOOL_OPEN_RE.findall(text))
    close_count = len(TOOL_CLOSE_RE.findall(text))

    for block in blocks:
        records.append(
            classify_tool_payload(
                payload=block.group(1),
                dataset=dataset,
                sample_index=sample_index,
                turn_index=turn_index,
                features_done=features_done,
                seen_signatures=seen_signatures,
            )
        )

    unmatched = max(open_count, close_count) - len(blocks)
    for _ in range(max(0, unmatched)):
        records.append(
            CallRecord(
                dataset=dataset,
                sample_index=sample_index,
                turn_index=turn_index,
                raw_payload="",
                call_kind="format_error",
                detail="unmatched_tool_call_tag",
            )
        )
    return records


def run_episode(
    *,
    llm: LLM,
    tokenizer: AutoTokenizer,
    sampling_params: SamplingParams,
    dataset_key: str,
    sample_index: int,
    data_str: str,
    lookback: int,
    horizon: int,
    max_steps: int,
) -> dict[str, Any]:
    timestamps, values = parse_time_series_string(data_str)
    ctx = {
        "timestamps": timestamps,
        "values": values,
        "features_done": False,
        "horizon": horizon,
    }
    history: list[str] = []
    prediction_str: str | None = None
    seen_signatures: set[str] = set()

    turn_records: list[dict[str, Any]] = []
    answer = None

    for turn_index in range(1, max_steps + 1):
        user_prompt = build_user_prompt(data_str, lookback, horizon, history, prediction_str)
        messages = [
            {"role": "system", "content": TIMESERIES_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tools=TIMESERIES_TOOL_SCHEMAS,
            add_generation_prompt=True,
            tokenize=False,
        )
        outputs = llm.generate([prompt_text], sampling_params, use_tqdm=False)
        out0 = outputs[0].outputs[0]
        text = out0.text
        answer = extract_answer(text)
        call_records = analyze_response_text(
            text=text,
            dataset=dataset_key,
            sample_index=sample_index,
            turn_index=turn_index,
            features_done=ctx["features_done"],
            seen_signatures=seen_signatures,
        )

        executed_valid_calls = 0
        for record in call_records:
            if record.call_kind != "valid" or not record.name:
                continue
            executed_valid_calls += 1
            signature = normalize_signature(record.name, record.arguments or {})
            seen_signatures.add(signature)
            tool_result = execute_local_tool(record.name, record.arguments or {}, ctx)
            if record.name == "predict_time_series" and not tool_result.startswith("ERROR"):
                prediction_str = tool_result
                history.append(
                    f"Model Prediction generated using {(record.arguments or {}).get('model_name', 'chronos2')}"
                )
            elif record.name in FEATURE_TOOLS:
                history.append(tool_result)

        turn_records.append(
            {
                "turn_index": turn_index,
                "assistant_text": text,
                "answer_present": bool(answer),
                "call_records": [asdict(record) for record in call_records],
                "executed_valid_calls": executed_valid_calls,
            }
        )

        if answer:
            break
        if not history and prediction_str is None:
            break

    return {
        "dataset": dataset_key,
        "sample_index": sample_index,
        "answer_present": bool(answer),
        "turns": turn_records,
    }


def aggregate_results(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    overall = Counter()
    by_dataset: dict[str, Counter] = defaultdict(Counter)
    detail_counters: dict[str, Counter] = defaultdict(Counter)

    for episode in episodes:
        dataset = episode["dataset"]
        overall["episodes"] += 1
        by_dataset[dataset]["episodes"] += 1

        for turn in episode["turns"]:
            overall["turns"] += 1
            by_dataset[dataset]["turns"] += 1
            if turn["answer_present"]:
                overall["answer_turns"] += 1
                by_dataset[dataset]["answer_turns"] += 1

            for call in turn["call_records"]:
                overall["tool_call_attempts"] += 1
                by_dataset[dataset]["tool_call_attempts"] += 1

                kind = call["call_kind"]
                if kind == "valid":
                    overall["valid_calls"] += 1
                    by_dataset[dataset]["valid_calls"] += 1
                    if call.get("redundant"):
                        overall["redundant_calls"] += 1
                        by_dataset[dataset]["redundant_calls"] += 1
                elif kind == "illegal":
                    overall["illegal_calls"] += 1
                    by_dataset[dataset]["illegal_calls"] += 1
                elif kind == "format_error":
                    overall["format_errors"] += 1
                    by_dataset[dataset]["format_errors"] += 1
                elif kind == "empty":
                    overall["empty_calls"] += 1
                    by_dataset[dataset]["empty_calls"] += 1

                detail = call.get("detail") or "unknown"
                detail_counters[f"{dataset}:{kind}"][detail] += 1
                detail_counters[f"overall:{kind}"][detail] += 1

        if episode["answer_present"]:
            overall["episodes_with_answer"] += 1
            by_dataset[dataset]["episodes_with_answer"] += 1

    def with_rates(counter: Counter) -> dict[str, Any]:
        attempts = int(counter["tool_call_attempts"])
        denom = attempts if attempts > 0 else 1
        return {
            "episodes": int(counter["episodes"]),
            "turns": int(counter["turns"]),
            "tool_call_attempts": attempts,
            "valid_calls": int(counter["valid_calls"]),
            "illegal_calls": int(counter["illegal_calls"]),
            "format_errors": int(counter["format_errors"]),
            "empty_calls": int(counter["empty_calls"]),
            "redundant_calls": int(counter["redundant_calls"]),
            "episodes_with_answer": int(counter["episodes_with_answer"]),
            "answer_success_rate": float(counter["episodes_with_answer"] / counter["episodes"]) if counter["episodes"] else 0.0,
            "illegal_call_rate": float(counter["illegal_calls"] / denom),
            "format_error_rate": float(counter["format_errors"] / denom),
            "empty_call_rate": float(counter["empty_calls"] / denom),
            "redundant_call_rate": float(counter["redundant_calls"] / denom),
            "valid_call_rate": float(counter["valid_calls"] / denom),
        }

    return {
        "overall": with_rates(overall),
        "by_dataset": {k: with_rates(v) for k, v in sorted(by_dataset.items())},
        "detail_breakdown": {k: dict(v) for k, v in sorted(detail_counters.items())},
        "episodes": episodes,
    }


def build_markdown(
    *,
    model: str,
    num_samples: int,
    max_steps: int,
    metric_bundle: dict[str, Any],
    elapsed_s: float,
) -> str:
    overall = metric_bundle["overall"]
    lines = [
        "# Cast-R1 Tool-Call Validity Benchmark",
        "",
        f"- Model: `{model}`",
        "- Protocol: same 3-turn Cast-R1 agent workflow as `benchmark_castr1.py`",
        "- Tool executor: local feature tools + deterministic local prediction stub for Turn-3 continuation",
        f"- Samples per dataset: `{num_samples}`",
        f"- Max steps: `{max_steps}`",
        f"- Elapsed time (s): `{elapsed_s:.2f}`",
        "- Rate denominator: total emitted tool-call attempts",
        "- Redundant calls are a subset of syntactically valid calls and therefore overlap with valid-call counts",
        "",
        "## Overall",
        "",
        f"- Episodes: `{overall['episodes']}`",
        f"- Tool-call attempts: `{overall['tool_call_attempts']}`",
        f"- Valid calls: `{overall['valid_calls']}` ({format_float(overall['valid_call_rate'])})",
        f"- Illegal call rate: `{format_float(overall['illegal_call_rate'])}` ({overall['illegal_calls']}/{overall['tool_call_attempts']})",
        f"- Format error rate: `{format_float(overall['format_error_rate'])}` ({overall['format_errors']}/{overall['tool_call_attempts']})",
        f"- Empty call rate: `{format_float(overall['empty_call_rate'])}` ({overall['empty_calls']}/{overall['tool_call_attempts']})",
        f"- Duplicate / redundant call rate: `{format_float(overall['redundant_call_rate'])}` ({overall['redundant_calls']}/{overall['tool_call_attempts']})",
        f"- Answer success rate: `{format_float(overall['answer_success_rate'])}` ({overall['episodes_with_answer']}/{overall['episodes']})",
        "",
        "## By Dataset",
        "",
        "| Dataset | Episodes | Attempts | Illegal | Format | Empty | Redundant | Valid |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for dataset, row in metric_bundle["by_dataset"].items():
        lines.append(
            f"| {dataset} | {row['episodes']} | {row['tool_call_attempts']} | "
            f"{format_float(row['illegal_call_rate'])} | {format_float(row['format_error_rate'])} | "
            f"{format_float(row['empty_call_rate'])} | {format_float(row['redundant_call_rate'])} | "
            f"{format_float(row['valid_call_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## Detail Breakdown",
            "",
            "### Overall Illegal Calls",
            "",
            "```json",
            json.dumps(metric_bundle["detail_breakdown"].get("overall:illegal", {}), indent=2, ensure_ascii=False),
            "```",
            "",
            "### Overall Format Errors",
            "",
            "```json",
            json.dumps(metric_bundle["detail_breakdown"].get("overall:format_error", {}), indent=2, ensure_ascii=False),
            "```",
            "",
            "### Overall Empty Calls",
            "",
            "```json",
            json.dumps(metric_bundle["detail_breakdown"].get("overall:empty", {}), indent=2, ensure_ascii=False),
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(PROJECT_ROOT / "models" / "Qwen3-8B"))
    ap.add_argument("--datasets", default="etth1,np,wind", help="comma-separated preset keys")
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--num-warmup", type=int, default=2)
    ap.add_argument("--sample-offset", type=int, default=0, help="starting sample index within each dataset")
    ap.add_argument("--max-steps", type=int, default=3)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", required=True)
    args = ap.parse_args()

    dataset_keys = [x.strip().lower() for x in args.datasets.split(",") if x.strip()]
    unknown = [x for x in dataset_keys if x not in DATASET_PRESETS]
    if unknown:
        raise SystemExit(f"Unknown dataset presets: {unknown}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        enforce_eager=False,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    episodes: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    for dataset_key in dataset_keys:
        preset = DATASET_PRESETS[dataset_key]
        total_needed = args.sample_offset + args.num_warmup + args.num_samples
        samples = load_samples(preset["path"], total_needed)
        warm_start = args.sample_offset
        eval_start = args.sample_offset + args.num_warmup
        warm_samples = samples[warm_start:eval_start]
        eval_samples = samples[eval_start : eval_start + args.num_samples]

        print(f"[{dataset_key}] warmup={len(warm_samples)} eval={len(eval_samples)}")
        for sample_text in warm_samples:
            run_episode(
                llm=llm,
                tokenizer=tokenizer,
                sampling_params=sampling_params,
                dataset_key=dataset_key,
                sample_index=-1,
                data_str=sample_text,
                lookback=preset["lookback"],
                horizon=preset["horizon"],
                max_steps=args.max_steps,
            )

        for sample_index, sample_text in enumerate(eval_samples):
            start = time.perf_counter()
            episode = run_episode(
                llm=llm,
                tokenizer=tokenizer,
                sampling_params=sampling_params,
                dataset_key=dataset_key,
                sample_index=sample_index,
                data_str=sample_text,
                lookback=preset["lookback"],
                horizon=preset["horizon"],
                max_steps=args.max_steps,
            )
            episodes.append(episode)

            attempts = sum(len(turn["call_records"]) for turn in episode["turns"])
            illegal = sum(1 for turn in episode["turns"] for call in turn["call_records"] if call["call_kind"] == "illegal")
            fmt = sum(1 for turn in episode["turns"] for call in turn["call_records"] if call["call_kind"] == "format_error")
            empty = sum(1 for turn in episode["turns"] for call in turn["call_records"] if call["call_kind"] == "empty")
            redundant = sum(
                1
                for turn in episode["turns"]
                for call in turn["call_records"]
                if call["call_kind"] == "valid" and call["redundant"]
            )
            print(
                f"[{dataset_key} {sample_index + 1:02d}/{len(eval_samples):02d}] "
                f"{time.perf_counter() - start:6.2f}s attempts={attempts} "
                f"illegal={illegal} format={fmt} empty={empty} redundant={redundant} "
                f"answer={'Y' if episode['answer_present'] else 'N'}"
            )

    elapsed = time.perf_counter() - t0
    metric_bundle = aggregate_results(episodes)
    output = {
        "config": {
            "model": args.model,
            "datasets": dataset_keys,
            "num_samples": args.num_samples,
            "num_warmup": args.num_warmup,
            "sample_offset": args.sample_offset,
            "max_steps": args.max_steps,
            "tp": args.tp,
            "temperature": args.temperature,
            "max_model_len": args.max_model_len,
            "max_tokens": args.max_tokens,
            "metric_denominator": "total_emitted_tool_call_attempts",
            "tool_executor": "local_feature_tools_plus_local_prediction_stub",
        },
        "elapsed_s": elapsed,
        "summary": {
            "overall": metric_bundle["overall"],
            "by_dataset": metric_bundle["by_dataset"],
            "detail_breakdown": metric_bundle["detail_breakdown"],
        },
        "episodes": metric_bundle["episodes"],
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    output_md.write_text(
        build_markdown(
            model=args.model,
            num_samples=args.num_samples,
            max_steps=args.max_steps,
            metric_bundle=metric_bundle,
            elapsed_s=elapsed,
        )
    )

    print(json.dumps(output["summary"]["overall"], indent=2, ensure_ascii=False))
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()
