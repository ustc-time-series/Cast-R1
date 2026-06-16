from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from recipe.time_series_forecast.models.itransformer.model import create_itransformer_model
from recipe.time_series_forecast.models.patchtst.model import create_patchtst_model
from recipe.time_series_forecast.config_utils import get_dataset_lengths


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "datasets" / "RL"
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "tmp" / "small-model-checkpoints"
DEFAULT_DATASETS = ("ETTH1", "ETTH2", "ETTM1", "ETTM2", "WIND", "NP", "PJM", "BE", "DE", "FR")
DEFAULT_MODELS = ("itransformer", "patchtst")


def _parse_series_values(text: str) -> list[float]:
    values = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        values.append(float(line.split()[-1]))
    return values


def load_parquet_window_data(
    parquet_path: str | Path,
    *,
    seq_len: int = 96,
    pred_len: int = 96,
) -> tuple[torch.Tensor, torch.Tensor]:
    dataframe = pd.read_parquet(parquet_path)
    inputs = []
    targets = []
    for row_idx, row in dataframe.iterrows():
        prompt = row["prompt"][0]["content"]
        ground_truth = row["reward_model"]["ground_truth"]
        input_values = _parse_series_values(prompt)
        target_values = _parse_series_values(ground_truth)
        if len(input_values) != seq_len or len(target_values) != pred_len:
            raise ValueError(
                f"{parquet_path} row {row_idx} has lengths "
                f"input={len(input_values)} target={len(target_values)}, expected {seq_len}/{pred_len}"
            )
        inputs.append(input_values)
        targets.append(target_values)

    input_tensor = torch.tensor(inputs, dtype=torch.float32).unsqueeze(-1)
    target_tensor = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)
    return input_tensor, target_tensor


class RLWindowDataset(Dataset):
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor):
        if len(inputs) != len(targets):
            raise ValueError(f"inputs and targets must have same length: {len(inputs)} != {len(targets)}")
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index]


@dataclass
class SmallModelTrainingConfig:
    model_name: str
    train_file: Path
    val_file: Path
    output_path: Path
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "auto"
    seed: int = 42
    seq_len: int = 96
    pred_len: int = 96
    enc_in: int = 1
    d_model: int | None = None
    n_heads: int | None = None
    e_layers: int | None = None
    d_ff: int | None = None
    dropout: float = 0.0
    patch_len: int = 16
    stride: int = 8
    num_workers: int = 0
    grad_clip: float = 1.0


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        try:
            import torch_npu  # noqa: F401

            if hasattr(torch, "npu") and torch.npu.is_available():
                return torch.device("npu:0")
        except Exception:
            pass
        return torch.device("cpu")

    if device.startswith("npu"):
        import torch_npu  # noqa: F401

    return torch.device(device)


def _model_config(config: SmallModelTrainingConfig) -> dict:
    if config.model_name == "itransformer":
        return {
            "seq_len": config.seq_len,
            "pred_len": config.pred_len,
            "enc_in": config.enc_in,
            "d_model": config.d_model or 128,
            "n_heads": config.n_heads or 8,
            "e_layers": config.e_layers or 2,
            "d_ff": config.d_ff or 128,
            "dropout": config.dropout,
        }
    if config.model_name == "patchtst":
        return {
            "seq_len": config.seq_len,
            "pred_len": config.pred_len,
            "enc_in": config.enc_in,
            "d_model": config.d_model or 512,
            "n_heads": config.n_heads or 2,
            "e_layers": config.e_layers or 1,
            "d_ff": config.d_ff or 2048,
            "dropout": config.dropout,
            "patch_len": config.patch_len,
            "stride": config.stride,
        }
    raise ValueError(f"Unsupported model_name={config.model_name}")


def build_model(config: SmallModelTrainingConfig) -> nn.Module:
    model_config = _model_config(config)
    if config.model_name == "itransformer":
        return create_itransformer_model(model_config)
    if config.model_name == "patchtst":
        return create_patchtst_model(model_config)
    raise ValueError(f"Unsupported model_name={config.model_name}")


def _make_loader(
    parquet_path: Path,
    *,
    batch_size: int,
    seq_len: int,
    pred_len: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    inputs, targets = load_parquet_window_data(parquet_path, seq_len=seq_len, pred_len=pred_len)
    dataset = RLWindowDataset(inputs, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def _mean_epoch_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            predictions = model(inputs)
            losses.append(float(criterion(predictions, targets).detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def train_and_save(config: SmallModelTrainingConfig) -> dict[str, float]:
    _set_seed(config.seed)
    config.train_file = Path(config.train_file)
    config.val_file = Path(config.val_file)
    config.output_path = Path(config.output_path)

    device = _resolve_device(config.device)
    train_loader = _make_loader(
        config.train_file,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = _make_loader(
        config.val_file,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = build_model(config).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    last_train_loss = float("nan")
    last_val_loss = float("nan")
    best_val_loss = float("inf")
    best_state_dict = None

    for epoch in range(config.epochs):
        model.train()
        train_losses = []
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        last_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        last_val_loss = _mean_epoch_loss(model, val_loader, criterion, device)
        if last_val_loss < best_val_loss:
            best_val_loss = last_val_loss
            best_state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        print(
            f"[{config.model_name}] epoch={epoch + 1}/{config.epochs} "
            f"train_loss={last_train_loss:.6f} val_loss={last_val_loss:.6f}",
            flush=True,
        )

    state_dict = best_state_dict or {key: value.detach().cpu() for key, value in model.state_dict().items()}
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    model_config = _model_config(config)
    metrics = {
        "train_loss": last_train_loss,
        "val_loss": last_val_loss,
        "best_val_loss": best_val_loss,
    }
    torch.save(
        {
            "state_dict": state_dict,
            "model_config": model_config,
            "metrics": metrics,
        },
        config.output_path,
    )
    with (config.output_path.parent / "config.json").open("w", encoding="utf-8") as config_file:
        json.dump(model_config, config_file, indent=2)
        config_file.write("\n")
    return metrics


def _split_csv(raw_value: str) -> tuple[str, ...]:
    return tuple(item.strip().upper() for item in raw_value.split(",") if item.strip())


def _resolve_dataset_dir(data_root: Path, dataset: str) -> Path:
    dataset_dir = data_root / dataset
    if dataset_dir.exists():
        return dataset_dir

    if dataset == "WIND":
        wind_dir = data_root / "Wind"
        if wind_dir.exists():
            return wind_dir

    return dataset_dir


def _iter_training_configs(args: argparse.Namespace) -> Iterable[SmallModelTrainingConfig]:
    datasets = _split_csv(args.datasets)
    models = tuple(item.strip().lower() for item in args.models.split(",") if item.strip())
    for dataset in datasets:
        dataset_dir = _resolve_dataset_dir(args.data_root, dataset)
        seq_len, pred_len = get_dataset_lengths(dataset)
        for model_name in models:
            yield SmallModelTrainingConfig(
                model_name=model_name,
                train_file=dataset_dir / "train.parquet",
                val_file=dataset_dir / "val.parquet",
                output_path=args.model_root / model_name / dataset / "checkpoint.pth",
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                device=args.device,
                seed=args.seed,
                seq_len=seq_len,
                pred_len=pred_len,
                num_workers=args.num_workers,
                grad_clip=args.grad_clip,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PatchTST/iTransformer from AgentRFT RL parquet windows.")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    args = parser.parse_args()

    for config in _iter_training_configs(args):
        print(f"Training {config.model_name} on {config.train_file.parent.name}")
        metrics = train_and_save(config)
        print(f"Saved {config.output_path} with metrics={metrics}")


if __name__ == "__main__":
    main()
