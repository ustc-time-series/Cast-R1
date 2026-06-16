from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = PROJECT_ROOT / "examples" / "time_series_forecast"
PUBLIC_EXAMPLE_FILES = [
    "examples/time_series_forecast/run_inference_benchmark.sh",
    "examples/time_series_forecast/run_qwen3-4B.sh",
    "examples/time_series_forecast/train_small_models_from_rl.sh",
]
BANNED_SNIPPETS = [
    "ASCEND_RT_VISIBLE_DEVICES",
    "ASCEND_VISIBLE_DEVICES",
    "NPU_VISIBLE_DEVICES",
    "npu-smi",
    "/root/my/npu_keeper",
    "/root/wyc/verl",
    "/root/zhj/AgentRFT",
    "/root/zyt/miniconda3",
]


def _read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_public_examples_are_trimmed() -> None:
    files = sorted(
        str(path.relative_to(PROJECT_ROOT))
        for path in EXAMPLES_DIR.rglob("*")
        if path.is_file()
    )
    assert files == PUBLIC_EXAMPLE_FILES


def test_public_examples_are_repo_relative_and_cuda_first() -> None:
    for relative_path in PUBLIC_EXAMPLE_FILES:
        content = _read_text(relative_path)
        assert "PROJECT_ROOT" in content
        assert "PYTHON_BIN" in content
        assert "CUDA_VISIBLE_DEVICES" in content
        assert 'DEVICE="${DEVICE:-cuda}"' in content
        for banned in BANNED_SNIPPETS:
            assert banned not in content


def test_model_server_launcher_is_cuda_first() -> None:
    content = _read_text("recipe/time_series_forecast/start_model_server.sh")
    assert "PROJECT_ROOT" in content
    assert "PYTHON_BIN" in content
    assert "CUDA_VISIBLE_DEVICES" in content
    assert 'DEVICE="${3:-${DEVICE:-cuda}}"' in content
    for banned in BANNED_SNIPPETS:
        assert banned not in content
