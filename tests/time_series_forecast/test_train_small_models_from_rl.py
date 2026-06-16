from argparse import Namespace
from pathlib import Path

from recipe.time_series_forecast.train_small_models_from_rl import _iter_training_configs


def test_wind_dataset_uses_existing_mixed_case_data_dir_and_uppercase_model_dir(tmp_path: Path):
    data_root = tmp_path / "data"
    model_root = tmp_path / "models"
    (data_root / "Wind").mkdir(parents=True)

    args = Namespace(
        datasets="WIND",
        models="itransformer",
        data_root=data_root,
        model_root=model_root,
        epochs=1,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        device="cpu",
        seed=42,
        num_workers=0,
        grad_clip=1.0,
    )

    config = next(iter(_iter_training_configs(args)))

    assert config.train_file == data_root / "Wind" / "train.parquet"
    assert config.val_file == data_root / "Wind" / "val.parquet"
    assert config.output_path == model_root / "itransformer" / "WIND" / "checkpoint.pth"
    assert config.seq_len == 96
    assert config.pred_len == 96
