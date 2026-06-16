#!/usr/bin/env python3
"""
Unified Time Series Prediction Service

A FastAPI service that provides time series prediction using multiple models:
- Chronos2: Foundation model for time series
- PatchTST: Patch-based Transformer
- iTransformer: Inverted Transformer

This service can run on a dedicated GPU, separate from the training framework.

Usage:
    # Start the server on CUDA GPU 0 (loads all available models)
    CUDA_VISIBLE_DEVICES=0 python model_server.py --port 8993 --device cuda

    # Or on another supported device type
    python model_server.py --port 8994 --device cpu
"""

import argparse
import os
import json
import sys
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
try:
    import torch_npu  # noqa: F401
    _TORCH_NPU_IMPORTED = True
except Exception:
    _TORCH_NPU_IMPORTED = False
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from recipe.time_series_forecast.config_utils import get_dataset_lengths, get_default_lengths


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Base path for models
MODELS_BASE_PATH = Path(__file__).resolve().parent / "models"

# Global model caches
_models: Dict[str, Any] = {
    "chronos2": None,
    "patchtst": {},
    "itransformer": {},
}

_configs: Dict[str, dict] = {}

# Model directories
_MODEL_DIRS = {
    "chronos2": MODELS_BASE_PATH / "chronos-2",
    "patchtst": MODELS_BASE_PATH / "patchtst",
    "itransformer": MODELS_BASE_PATH / "itransformer",
}

# Default lengths resolved from env/base.yaml
DEFAULT_LOOKBACK_WINDOW, DEFAULT_FORECAST_HORIZON = get_default_lengths()

# Active device resolved at runtime
_ACTIVE_DEVICE = "cpu"


def _normalize_device(device: str) -> str:
    if not device:
        return "cpu"
    device = device.strip().lower()
    if device == "gpu":
        return "cuda"
    return device


def _is_npu_available() -> bool:
    if not _TORCH_NPU_IMPORTED:
        return False
    if hasattr(torch, "npu") and hasattr(torch.npu, "is_available"):
        try:
            return torch.npu.is_available()
        except Exception:
            return False
    return False


def _resolve_device(device: str) -> str:
    device = _normalize_device(device)
    if device.startswith("npu"):
        if not _is_npu_available():
            print("[WARNING] NPU requested but torch_npu is not available. Falling back to CPU.")
            return "cpu"
        return "npu:0" if device == "npu" else device
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[WARNING] CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        return "cuda:0" if device == "cuda" else device
    if device == "cpu":
        return "cpu"
    try:
        torch.device(device)
        return device
    except Exception:
        print(f"[WARNING] Unknown device '{device}', falling back to CPU.")
        return "cpu"


def _set_device_context(device: str) -> None:
    if device.startswith("npu") and _is_npu_available():
        idx = 0
        if ":" in device:
            try:
                idx = int(device.split(":", 1)[1])
            except ValueError:
                idx = 0
        try:
            torch.npu.set_device(idx)
        except Exception as e:
            print(f"[WARNING] Failed to set NPU device to {device}: {e}")
    elif device.startswith("cuda") and torch.cuda.is_available():
        idx = 0
        if ":" in device:
            try:
                idx = int(device.split(":", 1)[1])
            except ValueError:
                idx = 0
        try:
            torch.cuda.set_device(idx)
        except Exception as e:
            print(f"[WARNING] Failed to set CUDA device to {device}: {e}")


def _safe_torch_load(checkpoint_path: Path, device: str):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        print(f"[WARNING] torch.load failed on {device}: {e}. Retrying on CPU.")
        try:
            return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(checkpoint_path, map_location="cpu")


def _get_health_device() -> str:
    if _ACTIVE_DEVICE.startswith("npu"):
        if _is_npu_available() and hasattr(torch, "npu"):
            try:
                return f"npu:{torch.npu.current_device()}"
            except Exception:
                return "npu"
        return "cpu"
    if _ACTIVE_DEVICE.startswith("cuda"):
        if torch.cuda.is_available():
            try:
                return f"cuda:{torch.cuda.current_device()}"
            except Exception:
                return "cuda"
        return "cpu"
    return "cpu"


# =============================================================================
# Request/Response Models
# =============================================================================

class PredictRequest(BaseModel):
    """Request model for prediction endpoint"""
    timestamps: List[str]  # List of timestamp strings
    values: List[float]    # List of time series values
    series_id: str = "series_0"
    prediction_length: int = DEFAULT_FORECAST_HORIZON
    model_name: str = "chronos2"  # Model to use
    dataset_name: Optional[str] = None  # Dataset-specific checkpoint, e.g. ETTH1/WIND/NP
    data_source: Optional[str] = None   # Backward-compatible alias for dataset_name


class PredictResponse(BaseModel):
    """Response model for prediction endpoint"""
    timestamps: List[str]   # Predicted timestamps
    values: List[float]     # Predicted values
    model_used: str
    dataset_used: Optional[str] = None
    status: str = "success"


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    models_loaded: Dict[str, bool]
    device: str


class ModelsInfoResponse(BaseModel):
    """Response model for models info"""
    available_models: List[str]
    models_status: Dict[str, Dict[str, Any]]


# =============================================================================
# Model Loading Functions
# =============================================================================

def _dataset_config_path(model_name: str, dataset_name: Optional[str]) -> Optional[Path]:
    model_dir = _MODEL_DIRS.get(model_name, MODELS_BASE_PATH / model_name)
    normalized_dataset = normalize_dataset_name(dataset_name)
    if not normalized_dataset:
        return None

    config_path = model_dir / normalized_dataset / "config.json"
    if config_path.exists():
        return config_path

    raw_dataset = str(dataset_name).strip()
    raw_config_path = model_dir / raw_dataset / "config.json"
    if raw_config_path.exists():
        return raw_config_path
    return None


def load_config(model_name: str, dataset_name: Optional[str] = None) -> dict:
    """Load model configuration from config.json."""
    dataset_config_path = _dataset_config_path(model_name, dataset_name)
    config_path = dataset_config_path or (_MODEL_DIRS.get(model_name, MODELS_BASE_PATH / model_name) / "config.json")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}

    if dataset_name is not None:
        dataset_lookback, dataset_horizon = get_dataset_lengths(dataset_name)
        if dataset_config_path is None:
            config["seq_len"] = dataset_lookback
            config["pred_len"] = dataset_horizon
        else:
            config.setdefault("seq_len", dataset_lookback)
            config.setdefault("pred_len", dataset_horizon)
    else:
        config.setdefault("seq_len", DEFAULT_LOOKBACK_WINDOW)
        config.setdefault("pred_len", DEFAULT_FORECAST_HORIZON)

    return config


def normalize_dataset_name(dataset_name: Optional[str]) -> Optional[str]:
    """Normalize dataset identifiers used by RL parquet rows and checkpoint folders."""
    if dataset_name is None:
        return None
    normalized = str(dataset_name).strip()
    if not normalized:
        return None
    return normalized.upper()


def _request_dataset_name(request: PredictRequest) -> Optional[str]:
    return normalize_dataset_name(request.dataset_name or request.data_source)


def _model_cache_key(dataset_name: Optional[str]) -> str:
    return normalize_dataset_name(dataset_name) or "__default__"


def resolve_checkpoint_path(model_name: str, dataset_name: Optional[str] = None) -> Path:
    """Resolve the checkpoint path for a model, preferring dataset-specific weights."""
    model_dir = _MODEL_DIRS.get(model_name, MODELS_BASE_PATH / model_name)
    normalized_dataset = normalize_dataset_name(dataset_name)
    if normalized_dataset:
        checkpoint_path = model_dir / normalized_dataset / "checkpoint.pth"
        if checkpoint_path.exists():
            return checkpoint_path

        raw_dataset = str(dataset_name).strip()
        raw_checkpoint_path = model_dir / raw_dataset / "checkpoint.pth"
        if raw_checkpoint_path.exists():
            return raw_checkpoint_path

        raise FileNotFoundError(
            f"{model_name} checkpoint not found for dataset {normalized_dataset}: {checkpoint_path}"
        )

    return model_dir / "checkpoint.pth"


def _load_model_state(model: torch.nn.Module, checkpoint_path: Path, device: str) -> None:
    checkpoint = _safe_torch_load(checkpoint_path, device)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)


def _is_tsl_native_config(config: dict) -> bool:
    return str(config.get("backend", "")).strip().lower() == "tsl_native"


def _ensure_tsl_library_path(config: dict) -> Path:
    tsl_path = Path(
        config.get("tsl_library_path")
        or os.environ.get("TSL_LIBRARY_PATH")
        or (PROJECT_ROOT / "Time-Series-Library")
    )
    if not tsl_path.exists():
        raise FileNotFoundError(f"Time-Series-Library path not found: {tsl_path}")
    tsl_path_str = str(tsl_path)
    if tsl_path_str not in sys.path:
        sys.path.insert(0, tsl_path_str)
    return tsl_path


def _time_features_for_tsl(timestamps: List[pd.Timestamp], freq: str) -> np.ndarray:
    """Match Time-Series-Library timeF features used by ETT minute/hour datasets."""
    index = pd.DatetimeIndex(timestamps)
    freq = str(freq or "h").lower()
    features = []
    if freq in {"t", "min", "minute", "15min"}:
        features.append(index.minute / 59.0 - 0.5)
    if freq in {"t", "min", "minute", "15min", "h", "hour"}:
        features.append(index.hour / 23.0 - 0.5)
    features.extend(
        [
            index.dayofweek / 6.0 - 0.5,
            (index.day - 1) / 30.0 - 0.5,
            (index.dayofyear - 1) / 365.0 - 0.5,
        ]
    )
    return np.vstack(features).T.astype(np.float32)


class TSLNativeForecastWrapper(torch.nn.Module):
    """Run Time-Series-Library models with the same global scaler used during TSL training."""

    def __init__(self, model: torch.nn.Module, config: dict):
        super().__init__()
        self.model = model
        self.config = dict(config)
        self.seq_len = int(config.get("seq_len", DEFAULT_LOOKBACK_WINDOW))
        self.pred_len = int(config.get("pred_len", DEFAULT_FORECAST_HORIZON))
        self.label_len = int(config.get("label_len", max(1, self.seq_len // 2)))
        self.freq = str(config.get("freq", "h"))
        self.scaler_mean = float(config["scaler_mean"])
        self.scaler_std = float(config["scaler_std"])

    def _build_mark(self, timestamps: Optional[List[pd.Timestamp]], batch_size: int, length: int, device) -> torch.Tensor:
        if timestamps is None:
            feature_count = 5 if str(self.freq).lower() in {"t", "min", "minute", "15min"} else 4
            return torch.zeros((batch_size, length, feature_count), dtype=torch.float32, device=device)

        features = _time_features_for_tsl(timestamps, self.freq)
        mark = torch.from_numpy(features).to(device=device, dtype=torch.float32)
        return mark.unsqueeze(0).repeat(batch_size, 1, 1)

    def forward(
        self,
        x_raw: torch.Tensor,
        timestamps: Optional[List[pd.Timestamp]] = None,
        prediction_length: Optional[int] = None,
    ) -> torch.Tensor:
        pred_len = prediction_length or self.pred_len
        x_scaled = (x_raw - self.scaler_mean) / self.scaler_std

        batch_size, _, n_vars = x_scaled.shape
        device = x_scaled.device
        x_mark_enc = self._build_mark(timestamps, batch_size, x_scaled.shape[1], device)
        dec_len = self.label_len + self.pred_len
        x_dec = torch.zeros((batch_size, dec_len, n_vars), dtype=x_scaled.dtype, device=device)
        x_mark_dec = torch.zeros((batch_size, dec_len, x_mark_enc.shape[-1]), dtype=x_scaled.dtype, device=device)

        out_scaled = self.model(x_scaled, x_mark_enc, x_dec, x_mark_dec)
        out_scaled = out_scaled[:, -pred_len:, :]
        return out_scaled * self.scaler_std + self.scaler_mean


def create_tsl_native_model(model_name: str, config: dict) -> torch.nn.Module:
    _ensure_tsl_library_path(config)
    if model_name == "patchtst":
        from models.PatchTST import Model as TSLModel
    elif model_name == "itransformer":
        from models.iTransformer import Model as TSLModel
    else:
        raise ValueError(f"Unsupported TSL native model: {model_name}")

    args = SimpleNamespace(
        task_name=config.get("task_name", "long_term_forecast"),
        seq_len=int(config.get("seq_len", DEFAULT_LOOKBACK_WINDOW)),
        label_len=int(config.get("label_len", max(1, int(config.get("seq_len", DEFAULT_LOOKBACK_WINDOW)) // 2))),
        pred_len=int(config.get("pred_len", DEFAULT_FORECAST_HORIZON)),
        d_model=int(config.get("d_model", 512)),
        dropout=float(config.get("dropout", 0.0)),
        factor=int(config.get("factor", 3)),
        n_heads=int(config.get("n_heads", 8)),
        d_ff=int(config.get("d_ff", 2048)),
        e_layers=int(config.get("e_layers", 1)),
        activation=config.get("activation", "gelu"),
        enc_in=int(config.get("enc_in", 1)),
        embed=config.get("embed", "timeF"),
        freq=config.get("freq", "h"),
    )
    return TSLNativeForecastWrapper(TSLModel(args).float(), config)


def load_chronos2(device: str = "cuda"):
    """Load the Chronos2 model using BaseChronosPipeline"""
    global _models
    device = _resolve_device(device)
    
    if _models["chronos2"] is None:
        model_dir = _MODEL_DIRS["chronos2"]
        if not model_dir.exists():
            print(f"[WARNING] Chronos2 model directory not found: {model_dir}")
            return None
        
        try:
            from chronos import BaseChronosPipeline
            print(f"Loading Chronos2 model from {model_dir} on device {device}...")
            try:
                _models["chronos2"] = BaseChronosPipeline.from_pretrained(str(model_dir), device_map=device)
            except Exception as e:
                if str(device).startswith("npu"):
                    print(f"[WARNING] Chronos2 load on NPU failed: {e}. Retrying on CPU.")
                    _models["chronos2"] = BaseChronosPipeline.from_pretrained(str(model_dir), device_map="cpu")
                else:
                    raise
            print("Chronos2 model loaded successfully!")
        except ImportError:
            print("[WARNING] chronos package is not installed. Skipping Chronos2 model.")
            return None
        except Exception as e:
            print(f"[WARNING] Failed to load Chronos2 model: {e}")
            return None
    
    return _models["chronos2"]


def load_patchtst(device: str = "cuda", dataset_name: Optional[str] = None):
    """Load the PatchTST model"""
    global _models, _configs
    device = _resolve_device(device)
    cache = _models.setdefault("patchtst", {})
    if cache is None or not isinstance(cache, dict):
        cache = {}
        _models["patchtst"] = cache
    cache_key = _model_cache_key(dataset_name)
    
    if cache_key not in cache:
        try:
            checkpoint_path = resolve_checkpoint_path("patchtst", dataset_name)
        except FileNotFoundError as e:
            print(f"[WARNING] {e}")
            return None
        
        if not checkpoint_path.exists():
            print(f"[WARNING] PatchTST checkpoint not found: {checkpoint_path}")
            return None
        
        try:
            # Load config
            config = load_config("patchtst", dataset_name=dataset_name)
            _configs["patchtst"] = config
            _configs[f"patchtst:{cache_key}"] = config

            if _is_tsl_native_config(config):
                model = create_tsl_native_model("patchtst", config)
            else:
                from recipe.time_series_forecast.models.patchtst.model import create_patchtst_model
                model = create_patchtst_model(config)
            
            # Load checkpoint
            state_target = model.model if isinstance(model, TSLNativeForecastWrapper) else model
            _load_model_state(state_target, checkpoint_path, device)
            
            model.to(device)
            model.eval()
            cache[cache_key] = model
            dataset_label = normalize_dataset_name(dataset_name) or "default"
            print(f"PatchTST model loaded successfully for {dataset_label} from {checkpoint_path}!")
            
        except Exception as e:
            print(f"[WARNING] Failed to load PatchTST model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    return cache[cache_key]


def load_itransformer(device: str = "npu", dataset_name: Optional[str] = None):
    """Load the iTransformer model"""
    global _models, _configs
    device = _resolve_device(device)
    cache = _models.setdefault("itransformer", {})
    if cache is None or not isinstance(cache, dict):
        cache = {}
        _models["itransformer"] = cache
    cache_key = _model_cache_key(dataset_name)
    
    if cache_key not in cache:
        try:
            checkpoint_path = resolve_checkpoint_path("itransformer", dataset_name)
        except FileNotFoundError as e:
            print(f"[WARNING] {e}")
            return None
        
        if not checkpoint_path.exists():
            print(f"[WARNING] iTransformer checkpoint not found: {checkpoint_path}")
            return None
        
        try:
            # Load config
            config = load_config("itransformer", dataset_name=dataset_name)
            _configs["itransformer"] = config
            _configs[f"itransformer:{cache_key}"] = config

            if _is_tsl_native_config(config):
                model = create_tsl_native_model("itransformer", config)
            else:
                from recipe.time_series_forecast.models.itransformer.model import create_itransformer_model
                model = create_itransformer_model(config)
            
            # Load checkpoint
            state_target = model.model if isinstance(model, TSLNativeForecastWrapper) else model
            _load_model_state(state_target, checkpoint_path, device)
            
            model.to(device)
            model.eval()
            cache[cache_key] = model
            dataset_label = normalize_dataset_name(dataset_name) or "default"
            print(f"iTransformer model loaded successfully for {dataset_label} from {checkpoint_path}!")
            
        except Exception as e:
            print(f"[WARNING] Failed to load iTransformer model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    return cache[cache_key]


def load_all_models(device: str = "cuda"):
    """Load all available models"""
    global _ACTIVE_DEVICE
    device = _resolve_device(device)
    _set_device_context(device)
    _ACTIVE_DEVICE = device
    print("=" * 60)
    print("Loading all available models...")
    print(f"Using device: {device}")
    print("=" * 60)
    
    load_chronos2(device)

    for model_name, loader in (("patchtst", load_patchtst), ("itransformer", load_itransformer)):
        default_checkpoint = resolve_checkpoint_path(model_name)
        if default_checkpoint.exists():
            loader(device)
        else:
            print(
                f"[INFO] {model_name} has no default checkpoint at {default_checkpoint}; "
                "dataset-specific checkpoints will be loaded lazily on request."
            )
    
    loaded = [
        name
        for name, model in _models.items()
        if (bool(model) if isinstance(model, dict) else model is not None)
    ]
    print("=" * 60)
    print(f"Models loaded: {loaded}")
    print("=" * 60)


# =============================================================================
# Prediction Functions
# =============================================================================

def predict_with_chronos2(request: PredictRequest) -> PredictResponse:
    """
    Generate predictions using Chronos2 with Non-stationary Transformer normalization.
    Uses BaseChronosPipeline.predict_quantiles (same as Time-Series-Library).
    """
    pipeline = _models["chronos2"]
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Chronos2 model not loaded")
    
    datetime_list = [pd.to_datetime(ts) for ts in request.timestamps]
    values = np.array(request.values, dtype=np.float32)
    
    # Convert to tensor: [batch=1, seq_len, n_vars=1]
    x_enc = torch.FloatTensor(values).unsqueeze(0).unsqueeze(-1)
    
    # Non-stationary Transformer Normalization
    means = x_enc.mean(1, keepdim=True).detach()
    x_enc = x_enc - means
    stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
    x_enc = x_enc / stdev
    
    # Reshape for Chronos: [batch, n_vars, seq_len]
    x_enc = x_enc.permute(0, 2, 1)
    
    # Predict using predict_quantiles
    quantiles, _ = pipeline.predict_quantiles(
        x_enc.cpu().numpy(),
        prediction_length=request.prediction_length,
        quantile_levels=[0.1, 0.5, 0.9]
    )
    
    # quantiles[0] shape: [batch, pred_len, num_quantiles]
    # Take median (index 1 in last dimension)
    dec_out = quantiles[0][:, :, 1]  # [batch, pred_len]
    dec_out = dec_out.unsqueeze(-1)  # [batch, pred_len, 1]
    
    # De-Normalization
    dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, request.prediction_length, 1)
    dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, request.prediction_length, 1)
    
    pred_values = dec_out.numpy().squeeze().tolist()
    
    # Generate timestamps
    freq = datetime_list[-1] - datetime_list[-2] if len(datetime_list) >= 2 else pd.Timedelta(hours=1)
    last_ts = datetime_list[-1]
    pred_timestamps = [(last_ts + freq * (i + 1)).strftime('%Y-%m-%d %H:%M:%S') 
                       for i in range(request.prediction_length)]
    
    return PredictResponse(
        timestamps=pred_timestamps,
        values=pred_values,
        model_used="chronos2",
        status="success"
    )


def predict_with_pytorch_model(request: PredictRequest, model_name: str) -> PredictResponse:
    """
    Generate predictions using PatchTST or iTransformer.
    
    Note: Models have Non-stationary Transformer normalization built-in (same as Time-Series-Library).
    Input raw data, output is in original scale.
    """
    dataset_name = _request_dataset_name(request)
    if model_name == "patchtst":
        model = load_patchtst(_ACTIVE_DEVICE, dataset_name=dataset_name)
    elif model_name == "itransformer":
        model = load_itransformer(_ACTIVE_DEVICE, dataset_name=dataset_name)
    else:
        model = None

    if model is None:
        dataset_msg = f" for dataset {dataset_name}" if dataset_name else ""
        raise HTTPException(status_code=503, detail=f"{model_name} model not loaded{dataset_msg}")
    
    device = next(model.parameters()).device
    
    # Prepare data
    values = np.array(request.values, dtype=np.float32)
    datetime_list = [pd.to_datetime(ts) for ts in request.timestamps]
    freq = datetime_list[-1] - datetime_list[-2] if len(datetime_list) >= 2 else pd.Timedelta(hours=1)
    
    # Prepare input tensor: [batch, seq_len, n_vars]
    input_tensor = torch.FloatTensor(values).unsqueeze(0).unsqueeze(-1).to(device)
    
    # Predict. TSL-native models additionally need timestamps for timeF features
    # and handle the global TSL dataset scaler inside the wrapper.
    with torch.no_grad():
        if isinstance(model, TSLNativeForecastWrapper):
            predictions = model(
                input_tensor,
                timestamps=datetime_list,
                prediction_length=request.prediction_length,
            )
        else:
            predictions = model(input_tensor)
    
    # Convert to numpy
    pred_values = predictions.cpu().numpy().squeeze()
    
    # Ensure correct length
    if len(pred_values) > request.prediction_length:
        pred_values = pred_values[:request.prediction_length]
    elif len(pred_values) < request.prediction_length:
        pad_len = request.prediction_length - len(pred_values)
        pred_values = np.concatenate([pred_values, [pred_values[-1]] * pad_len])
    
    # Generate timestamps
    last_ts = datetime_list[-1]
    pred_timestamps = [(last_ts + freq * (i + 1)).strftime('%Y-%m-%d %H:%M:%S') 
                       for i in range(request.prediction_length)]
    
    return PredictResponse(
        timestamps=pred_timestamps,
        values=pred_values.tolist(),
        model_used=model_name,
        dataset_used=dataset_name,
        status="success"
    )


# =============================================================================
# FastAPI App
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for model loading"""
    device = os.environ.get("MODEL_DEVICE", "cuda")
    load_all_models(device)
    yield


app = FastAPI(
    title="Time Series Prediction Service",
    description="Unified time series prediction service supporting Chronos2, PatchTST, and iTransformer",
    version="2.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        models_loaded={
            name: (bool(model) if isinstance(model, dict) else model is not None)
            for name, model in _models.items()
        },
        device=_get_health_device()
    )


@app.get("/models", response_model=ModelsInfoResponse)
async def models_info():
    """Get information about available models"""
    models_status = {}
    for name, model in _models.items():
        loaded_datasets = sorted(model.keys()) if isinstance(model, dict) else []
        checkpoint_base = _MODEL_DIRS.get(name, MODELS_BASE_PATH / name)
        dataset_checkpoints = []
        if name in {"patchtst", "itransformer"} and checkpoint_base.exists():
            dataset_checkpoints = sorted(
                path.parent.name for path in checkpoint_base.glob("*/checkpoint.pth")
            )
        models_status[name] = {
            "loaded": bool(model) if isinstance(model, dict) else model is not None,
            "loaded_datasets": loaded_datasets,
            "config": _configs.get(name, {}),
            "checkpoint_exists": (checkpoint_base / "checkpoint.pth").exists()
                                 if name != "chronos2" else (_MODEL_DIRS["chronos2"] / "model.safetensors").exists(),
            "dataset_checkpoints": dataset_checkpoints,
        }
    
    return ModelsInfoResponse(
        available_models=["chronos2", "patchtst", "itransformer"],
        models_status=models_status
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Generate time series predictions using the specified model.
    
    Args:
        request: PredictRequest containing timestamps, values, model_name, and prediction parameters
        
    Returns:
        PredictResponse with predicted timestamps and values
    """
    # Validate input
    if len(request.timestamps) != len(request.values):
        raise HTTPException(
            status_code=400, 
            detail="timestamps and values must have the same length"
        )
    
    if len(request.values) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 data points are required"
        )
    
    model_name = request.model_name.lower().strip()
    
    try:
        if model_name == "chronos2":
            return predict_with_chronos2(request)
        elif model_name in ["patchtst", "itransformer"]:
            return predict_with_pytorch_model(request, model_name)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {model_name}. Available: chronos2, patchtst, itransformer"
            )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[ERROR] Prediction failed: {error_msg}")
        print(f"[ERROR] Traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


def main():
    parser = argparse.ArgumentParser(description="Time Series Prediction Service")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8993, help="Port to bind")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda, cpu, npu, npu:0, etc.)")
    args = parser.parse_args()
    
    # Set device via environment variable
    os.environ["MODEL_DEVICE"] = args.device
    
    resolved_device = _resolve_device(args.device)
    print(f"Starting Time Series Prediction Service on {args.host}:{args.port} with device {resolved_device}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
