import json
import re
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd


_ANALYSIS_HEADER = "### Analysis History"
_PREDICTION_HEADER = "### Model Predictions"
_PLACEHOLDER_ANALYSIS = "No previous analysis performed."
_PLACEHOLDER_PREDICTION = "No predictions available yet. Call predict_time_series to generate forecasts."
_ANSWER_RE = re.compile(r"<answer>.*?</answer>", re.DOTALL | re.IGNORECASE)
_TOOL_MARKUP_RE = re.compile(r"</?(?:tool_call|tools)\b", re.IGNORECASE)


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


def has_valid_terminal_answer(records: list[dict]) -> bool:
    if not records:
        return False

    terminal_record = max(records, key=lambda record: int(record.get("step_index", -1)))
    response_text = terminal_record.get("response_text", "") or ""
    return bool(_ANSWER_RE.search(response_text)) and not bool(_TOOL_MARKUP_RE.search(response_text))


def trajectory_records_to_messages(records: list[dict]) -> list[dict]:
    if not records:
        return []

    ordered_records = sorted(records, key=lambda record: record["step_index"])
    first_prompt = ordered_records[0]["raw_prompt"]
    messages = [dict(message) for message in first_prompt]

    for idx, record in enumerate(ordered_records):
        response_text = record.get("response_text", "").strip()
        if response_text:
            messages.append({"role": "assistant", "content": response_text})

        if idx + 1 >= len(ordered_records):
            continue

        current_prompt = ordered_records[idx]["raw_prompt"][-1]["content"]
        next_prompt = ordered_records[idx + 1]["raw_prompt"][-1]["content"]
        observation = _derive_tool_observation(current_prompt, next_prompt)
        if observation:
            messages.append({"role": "tool", "content": observation})

    return messages


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


def build_multiturn_sft_rows(
    paths: Iterable[str | Path],
    min_final_score: float | None = None,
    min_keep_count: int = 0,
) -> list[dict]:
    grouped_records = load_step_records(paths)
    rows = []
    for trajectory_uid, steps in grouped_records.items():
        if not has_valid_terminal_answer(steps):
            continue

        rows.append(
            {
                "trajectory_uid": trajectory_uid,
                "sample_index": steps[0].get("sample_index"),
                "rollout_n": steps[0].get("rollout_n"),
                "final_score": steps[0].get("final_score"),
                "ground_truth": steps[0].get("ground_truth"),
                "messages": trajectory_records_to_messages(steps),
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


def write_multiturn_sft_parquet(
    paths: Iterable[str | Path],
    output_path: str | Path,
    min_final_score: float | None = None,
    min_keep_count: int = 0,
) -> None:
    rows = build_multiturn_sft_rows(paths, min_final_score=min_final_score, min_keep_count=min_keep_count)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path)


def _derive_tool_observation(current_prompt: str, next_prompt: str) -> str | None:
    current_sections = _parse_prompt_sections(current_prompt)
    next_sections = _parse_prompt_sections(next_prompt)

    current_prediction = current_sections["prediction"]
    next_prediction = next_sections["prediction"]
    if next_prediction and next_prediction != current_prediction:
        return next_prediction

    current_history = current_sections["history_lines"]
    next_history = next_sections["history_lines"]
    history_delta = [line for line in next_history if line not in current_history]
    if history_delta:
        return "\n".join(history_delta)

    return None


def _parse_prompt_sections(prompt_text: str) -> dict:
    analysis_text = _extract_section(prompt_text, _ANALYSIS_HEADER)
    prediction_text = _extract_section(prompt_text, _PREDICTION_HEADER)

    history_lines = _normalize_history_lines(analysis_text)
    prediction = _normalize_prediction_text(prediction_text)

    return {"history_lines": history_lines, "prediction": prediction}


def _extract_section(prompt_text: str, header: str) -> str:
    if header not in prompt_text:
        return ""

    section = prompt_text.split(header, 1)[1]
    section = section.lstrip()
    next_header_index = section.find("\n### ")
    if next_header_index != -1:
        section = section[:next_header_index]
    return section.strip()


def _normalize_history_lines(history_text: str) -> list[str]:
    lines = [line.strip() for line in history_text.splitlines() if line.strip()]
    if not lines or lines == [_PLACEHOLDER_ANALYSIS]:
        return []
    return lines


def _normalize_prediction_text(prediction_text: str) -> str:
    prediction_text = prediction_text.strip()
    if not prediction_text or prediction_text == _PLACEHOLDER_PREDICTION:
        return ""
    return prediction_text


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convert time-series trajectory JSONL files to verl multiturn SFT parquet.")
    parser.add_argument("--input_paths", nargs="+", required=True, help="One or more trajectory JSONL files.")
    parser.add_argument("--output_path", required=True, help="Output parquet path.")
    parser.add_argument("--min_final_score", type=float, default=None)
    parser.add_argument("--min_keep_count", type=int, default=0)
    args = parser.parse_args()

    write_multiturn_sft_parquet(
        args.input_paths,
        args.output_path,
        min_final_score=args.min_final_score,
        min_keep_count=args.min_keep_count,
    )
    print(f"Wrote multiturn SFT parquet to {args.output_path}")


if __name__ == "__main__":
    main()
