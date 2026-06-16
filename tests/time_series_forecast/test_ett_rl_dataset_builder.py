from __future__ import annotations

from typing import Any

from recipe.time_series_forecast import ett_rl_dataset_builder
from recipe.time_series_forecast import etth1_raw_dataset_builder as base_builder


class _MockResponse:
    def __init__(self, values: list[float]):
        self._values = values

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, list[float]]:
        return {"values": self._values}


def test_get_teacher_mse_includes_dataset_name_in_payload(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, json: dict[str, Any], timeout: int) -> _MockResponse:
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _MockResponse([1.0, 2.0])

    monkeypatch.setattr(base_builder.requests, "post", fake_post)

    mse = base_builder.get_teacher_mse(
        timestamps=["2016-07-01 00:00:00", "2016-07-01 00:15:00"],
        values=[0.5, 0.7],
        ground_truth=[1.0, 2.0],
        prediction_length=2,
        url="http://127.0.0.1:8993/predict",
        model_name="itransformer",
        dataset_name="ETTM2",
    )

    assert mse == 0.0
    assert captured["url"] == "http://127.0.0.1:8993/predict"
    assert captured["timeout"] == 60
    assert captured["json"]["model_name"] == "itransformer"
    assert captured["json"]["dataset_name"] == "ETTM2"


def test_make_teacher_mse_fn_passes_dataset_name(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_get_teacher_mse(
        timestamps: list[str],
        values: list[float],
        ground_truth: list[float],
        prediction_length: int,
        *,
        url: str,
        model_name: str,
        dataset_name: str | None = None,
    ) -> float:
        captured["timestamps"] = timestamps
        captured["values"] = values
        captured["ground_truth"] = ground_truth
        captured["prediction_length"] = prediction_length
        captured["url"] = url
        captured["model_name"] = model_name
        captured["dataset_name"] = dataset_name
        return 1.23

    monkeypatch.setattr(base_builder, "get_teacher_mse", fake_get_teacher_mse)

    teacher_mse_fn = ett_rl_dataset_builder._make_teacher_mse_fn(
        "http://127.0.0.1:8993/predict",
        "itransformer",
        "ETTM2",
    )
    mse = teacher_mse_fn(
        ["2016-07-01 00:00:00", "2016-07-01 00:15:00"],
        [0.5, 0.7],
        [1.0, 2.0],
        2,
    )

    assert mse == 1.23
    assert captured["url"] == "http://127.0.0.1:8993/predict"
    assert captured["model_name"] == "itransformer"
    assert captured["dataset_name"] == "ETTM2"
