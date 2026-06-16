# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from urllib.parse import urlparse
from pydantic import BaseModel
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, peak_widths
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.decomposition import PCA
from collections import Counter

from recipe.time_series_forecast.config_utils import get_default_lengths

try:
    import ruptures as rpt
except ImportError:
    rpt = None


# Model service configuration
# Set MODEL_SERVICE_URL environment variable to use remote service
# Default: http://localhost:8994
# This unified service supports: chronos2, patchtst, itransformer
_MODEL_SERVICE_URL = os.environ.get("MODEL_SERVICE_URL", "http://localhost:8993")

# Legacy support: CHRONOS_SERVICE_URL (deprecated, use MODEL_SERVICE_URL instead)
_CHRONOS_SERVICE_URL = os.environ.get("CHRONOS_SERVICE_URL", _MODEL_SERVICE_URL)

_MODEL_SERVICE_TIMEOUT = float(
    os.environ.get("MODEL_SERVICE_TIMEOUT", os.environ.get("CHRONOS_SERVICE_TIMEOUT", "300.0"))
)

# Global HTTP client cache for async requests
_httpx_client_cache = {}

# Default lengths resolved from env/base.yaml
DEFAULT_LOOKBACK_WINDOW, DEFAULT_FORECAST_HORIZON = get_default_lengths()


def _should_trust_env_for_service_url(service_url: str) -> bool:
    """Return whether httpx should honor env proxy settings for the given service URL."""
    hostname = urlparse(service_url).hostname
    return hostname not in {"localhost", "127.0.0.1", "::1"}


def _get_httpx_client(service_url: str):
    """Get or create an httpx async client for a service URL."""
    global _httpx_client_cache
    trust_env = _should_trust_env_for_service_url(service_url)
    cache_key = ("trust_env", trust_env, "timeout", _MODEL_SERVICE_TIMEOUT)
    if cache_key not in _httpx_client_cache:
        import httpx

        _httpx_client_cache[cache_key] = httpx.AsyncClient(timeout=_MODEL_SERVICE_TIMEOUT, trust_env=trust_env)
    return _httpx_client_cache[cache_key]


# Supported prediction models
SUPPORTED_MODELS = ["chronos2", "arima", "patchtst", "itransformer"]


def _normalize_dataset_name(dataset_name: Optional[str]) -> Optional[str]:
    if dataset_name is None:
        return None
    normalized = str(dataset_name).strip()
    if not normalized:
        return None
    return normalized.upper()


def _resolve_dataset_name(
    context_df: pd.DataFrame,
    dataset_name: Optional[str] = None,
    data_source: Optional[str] = None,
) -> Optional[str]:
    return _normalize_dataset_name(
        dataset_name
        or data_source
        or context_df.attrs.get("dataset_name")
        or context_df.attrs.get("data_source")
    )


async def predict_time_series_async(
    context_df: pd.DataFrame, 
    prediction_length: int = DEFAULT_FORECAST_HORIZON,
    model_name: str = "chronos2",
    dataset_name: Optional[str] = None,
    data_source: Optional[str] = None,
) -> pd.DataFrame:
    """
    Predict time series using the specified model (async version).
    
    This is the main entry point for time series prediction. It dispatches
    to the appropriate model based on the model_name parameter.
    
    Args:
        context_df: DataFrame containing historical data.
                   Must contain columns: 'id', 'timestamp', 'target'.
        prediction_length: Number of steps to forecast (default: DEFAULT_FORECAST_HORIZON)
        model_name: Name of the model to use. Supported: "chronos2", "arima", "patchtst", "itransformer"
    
    Returns:
        DataFrame containing predictions with 'timestamp' and 'target_0.5' columns.
    
    Example:
        >>> context_df = parse_time_series_to_dataframe(data_str)
        >>> pred_df = await predict_time_series_async(context_df, model_name="arima")
    """
    model_name = model_name.lower().strip()
    
    resolved_dataset_name = _resolve_dataset_name(context_df, dataset_name=dataset_name, data_source=data_source)

    if model_name == "chronos2":
        return await predict_with_chronos_async(context_df, prediction_length)
    elif model_name == "arima":
        return await predict_with_arima_async(context_df, prediction_length)
    elif model_name == "patchtst":
        return await predict_with_patchtst_async(
            context_df,
            prediction_length,
            dataset_name=resolved_dataset_name,
        )
    elif model_name == "itransformer":
        return await predict_with_itransformer_async(
            context_df,
            prediction_length,
            dataset_name=resolved_dataset_name,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}. Supported models: {SUPPORTED_MODELS}")


async def predict_with_chronos_async(
    context_df: pd.DataFrame,
    prediction_length: int = DEFAULT_FORECAST_HORIZON,
) -> pd.DataFrame:
    """
    Predict time series using Chronos2 service (async version).
    
    This function calls a remote model service to generate time series forecasts.
    The service should be started separately with:
        CUDA_VISIBLE_DEVICES=0 python model_server.py --port 8993 --device cuda
    
    Args:
        context_df: DataFrame containing historical data.
                   Must contain columns: 'id', 'timestamp', 'target'.
        prediction_length: Number of steps to forecast (default: DEFAULT_FORECAST_HORIZON)
    
    Returns:
        DataFrame containing predictions with 'timestamp' and 'target_0.5' columns.
    """
    import httpx
    
    client = _get_httpx_client(_CHRONOS_SERVICE_URL)
    
    # Prepare request data
    timestamps = []
    for ts in context_df['timestamp']:
        if isinstance(ts, pd.Timestamp):
            timestamps.append(ts.strftime('%Y-%m-%d %H:%M:%S'))
        else:
            timestamps.append(str(ts))
    
    values = context_df['target'].tolist()
    series_id = context_df['id'].iloc[0] if 'id' in context_df.columns else "series_0"
    
    request_data = {
        "timestamps": timestamps,
        "values": values,
        "series_id": series_id,
        "prediction_length": prediction_length
    }
    
    # Call the service
    try:
        response = await client.post(f"{_CHRONOS_SERVICE_URL}/predict", json=request_data)
        response.raise_for_status()
        result = response.json()
        
        # Convert response to DataFrame
        pred_df = pd.DataFrame({
            'timestamp': [pd.to_datetime(ts) for ts in result['timestamps']],
            'target_0.5': result['values']
        })
        
        return pred_df
        
    except httpx.HTTPError as e:
        raise RuntimeError(f"Chronos2 service error: {e}. Make sure the service is running at {_CHRONOS_SERVICE_URL}")


async def predict_with_arima_async(
    context_df: pd.DataFrame,
    prediction_length: int = DEFAULT_FORECAST_HORIZON,
) -> pd.DataFrame:
    """
    Predict time series using ARIMA model (async version).
    
    Uses auto-ARIMA to automatically select the best (p, d, q) parameters.
    This is a local model that doesn't require external services.
    
    Args:
        context_df: DataFrame containing historical data.
                   Must contain columns: 'id', 'timestamp', 'target'.
        prediction_length: Number of steps to forecast (default: DEFAULT_FORECAST_HORIZON)
    
    Returns:
        DataFrame containing predictions with 'timestamp' and 'target_0.5' columns.
    """
    import asyncio
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    import warnings
    
    # Run ARIMA in executor to avoid blocking
    loop = asyncio.get_event_loop()
    
    def _fit_and_predict():
        values = context_df['target'].values
        timestamps = context_df['timestamp'].tolist()
        
        # Infer frequency from timestamps
        if len(timestamps) >= 2:
            freq = timestamps[1] - timestamps[0]
        else:
            freq = pd.Timedelta(hours=1)
        
        # Determine differencing order (d) using ADF test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                adf_result = adfuller(values, maxlag=min(10, len(values) // 3))
                # If p-value > 0.05, series is non-stationary, need differencing
                d = 0 if adf_result[1] < 0.05 else 1
            except Exception:
                d = 1  # Default to d=1 if ADF test fails
        
        # Try different ARIMA configurations and select the best one
        best_model = None
        best_aic = float('inf')
        
        # Common ARIMA configurations to try
        configs = [
            (1, d, 1), (2, d, 1), (1, d, 2), (2, d, 2),
            (3, d, 1), (1, d, 3), (0, d, 1), (1, d, 0),
            (5, d, 0), (0, d, 5),  # Pure AR and MA models
        ]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for p, d_val, q in configs:
                try:
                    model = ARIMA(values, order=(p, d_val, q))
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_model = fitted
                except Exception:
                    continue
        
        # If no model worked, use a simple configuration
        if best_model is None:
            try:
                model = ARIMA(values, order=(1, 1, 1))
                best_model = model.fit()
            except Exception:
                # Last resort: use naive forecast (repeat last value)
                last_value = values[-1]
                pred_values = [last_value] * prediction_length
                pred_timestamps = [timestamps[-1] + freq * (i + 1) for i in range(prediction_length)]
                return pd.DataFrame({
                    'timestamp': pred_timestamps,
                    'target_0.5': pred_values
                })
        
        # Generate predictions
        forecast = best_model.forecast(steps=prediction_length)
        
        # Generate future timestamps
        last_timestamp = timestamps[-1]
        pred_timestamps = [last_timestamp + freq * (i + 1) for i in range(prediction_length)]
        
        pred_df = pd.DataFrame({
            'timestamp': pred_timestamps,
            'target_0.5': forecast.values if hasattr(forecast, 'values') else forecast
        })
        
        return pred_df
    
    return await loop.run_in_executor(None, _fit_and_predict)


async def predict_with_patchtst_async(
    context_df: pd.DataFrame,
    prediction_length: int = DEFAULT_FORECAST_HORIZON,
    dataset_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Predict time series using PatchTST model via the unified model service (async version).
    
    PatchTST divides time series into patches and applies transformer attention.
    Good for capturing local patterns and long-range dependencies.
    
    Note: Model has Non-stationary Transformer normalization built-in.
    Input raw data, output is in original scale.
    
    Args:
        context_df: DataFrame containing historical data.
                   Must contain columns: 'id', 'timestamp', 'target'.
        prediction_length: Number of steps to forecast (default: DEFAULT_FORECAST_HORIZON)
    
    Returns:
        DataFrame containing predictions with 'timestamp' and 'target_0.5' columns.
    """
    import httpx
    
    client = _get_httpx_client(_MODEL_SERVICE_URL)
    
    # Prepare request data
    timestamps = []
    for ts in context_df['timestamp']:
        if isinstance(ts, pd.Timestamp):
            timestamps.append(ts.strftime('%Y-%m-%d %H:%M:%S'))
        else:
            timestamps.append(str(ts))
    
    values = context_df['target'].tolist()
    series_id = context_df['id'].iloc[0] if 'id' in context_df.columns else "series_0"
    
    request_data = {
        "timestamps": timestamps,
        "values": values,
        "series_id": series_id,
        "prediction_length": prediction_length,
        "model_name": "patchtst"
    }
    if dataset_name:
        request_data["dataset_name"] = dataset_name
    
    # Call the unified model service
    try:
        response = await client.post(f"{_MODEL_SERVICE_URL}/predict", json=request_data)
        response.raise_for_status()
        result = response.json()
        
        # Convert response to DataFrame
        pred_df = pd.DataFrame({
            'timestamp': [pd.to_datetime(ts) for ts in result['timestamps']],
            'target_0.5': result['values']
        })
        
        return pred_df
        
    except httpx.HTTPError as e:
        raise RuntimeError(f"PatchTST service error: {e}. Make sure the model service is running at {_MODEL_SERVICE_URL}")


async def predict_with_itransformer_async(
    context_df: pd.DataFrame,
    prediction_length: int = DEFAULT_FORECAST_HORIZON,
    dataset_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Predict time series using iTransformer model via the unified model service (async version).
    
    iTransformer applies attention on the variate dimension instead of temporal dimension.
    Good for multivariate time series with inter-variate correlations.
    
    Note: Model has Non-stationary Transformer normalization built-in.
    Input raw data, output is in original scale.
    
    Args:
        context_df: DataFrame containing historical data.
                   Must contain columns: 'id', 'timestamp', 'target'.
        prediction_length: Number of steps to forecast (default: DEFAULT_FORECAST_HORIZON)
    
    Returns:
        DataFrame containing predictions with 'timestamp' and 'target_0.5' columns.
    """
    import httpx
    
    client = _get_httpx_client(_MODEL_SERVICE_URL)
    
    # Prepare request data
    timestamps = []
    for ts in context_df['timestamp']:
        if isinstance(ts, pd.Timestamp):
            timestamps.append(ts.strftime('%Y-%m-%d %H:%M:%S'))
        else:
            timestamps.append(str(ts))
    
    values = context_df['target'].tolist()
    series_id = context_df['id'].iloc[0] if 'id' in context_df.columns else "series_0"
    
    request_data = {
        "timestamps": timestamps,
        "values": values,
        "series_id": series_id,
        "prediction_length": prediction_length,
        "model_name": "itransformer"
    }
    if dataset_name:
        request_data["dataset_name"] = dataset_name
    
    # Call the unified model service
    try:
        response = await client.post(f"{_MODEL_SERVICE_URL}/predict", json=request_data)
        response.raise_for_status()
        result = response.json()
        
        # Convert response to DataFrame
        pred_df = pd.DataFrame({
            'timestamp': [pd.to_datetime(ts) for ts in result['timestamps']],
            'target_0.5': result['values']
        })
        
        return pred_df
        
    except httpx.HTTPError as e:
        raise RuntimeError(f"iTransformer service error: {e}. Make sure the model service is running at {_MODEL_SERVICE_URL}")


def predict_time_series(
    context_df: pd.DataFrame,
    prediction_length: int = DEFAULT_FORECAST_HORIZON,
) -> pd.DataFrame:
    """
    Predict time series using Chronos2 service (sync version).
    
    This function calls a remote model service to generate time series forecasts.
    The service should be started separately with:
        CUDA_VISIBLE_DEVICES=0 python model_server.py --port 8993 --device cuda
    
    Args:
        context_df: DataFrame containing historical data.
                   Must contain columns: 'id', 'timestamp', 'target'.
        prediction_length: Number of steps to forecast (default: DEFAULT_FORECAST_HORIZON)
    
    Returns:
        DataFrame containing predictions with 'timestamp' and 'target_0.5' columns.
    
    Example:
        >>> context_df = parse_time_series_to_dataframe(data_str)
        >>> pred_df = predict_time_series(context_df)
    """
    import requests
    
    # Prepare request data
    timestamps = []
    for ts in context_df['timestamp']:
        if isinstance(ts, pd.Timestamp):
            timestamps.append(ts.strftime('%Y-%m-%d %H:%M:%S'))
        else:
            timestamps.append(str(ts))
    
    values = context_df['target'].tolist()
    series_id = context_df['id'].iloc[0] if 'id' in context_df.columns else "series_0"
    
    request_data = {
        "timestamps": timestamps,
        "values": values,
        "series_id": series_id,
        "prediction_length": prediction_length
    }
    
    # Call the service
    try:
        response = requests.post(
            f"{_CHRONOS_SERVICE_URL}/predict", 
            json=request_data,
            timeout=60.0
        )
        response.raise_for_status()
        result = response.json()
        
        # Convert response to DataFrame
        pred_df = pd.DataFrame({
            'timestamp': [pd.to_datetime(ts) for ts in result['timestamps']],
            'target_0.5': result['values']
        })
        
        return pred_df
        
    except requests.RequestException as e:
        raise RuntimeError(f"Chronos2 service error: {e}. Make sure the service is running at {_CHRONOS_SERVICE_URL}")


def parse_time_series_string(data_str: str) -> Tuple[List[str], List[float]]:
    """
    Parse time series string into timestamps and values.
    
    Expected format:
        "2017-05-01 00:00:00 11.588\n2017-05-01 01:00:00 10.918\n..."
    
    Args:
        data_str: Time series data as a string with format "timestamp value\n..."
    
    Returns:
        Tuple of (timestamps_list, values_list)
    
    Example:
        >>> timestamps, values = parse_time_series_string("2017-05-01 00:00:00 11.588\\n2017-05-01 01:00:00 10.918")
        >>> print(timestamps)
        ['2017-05-01 00:00:00', '2017-05-01 01:00:00']
        >>> print(values)
        [11.588, 10.918]
    """
    timestamps = []
    values = []
    
    lines = data_str.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Pattern: datetime (YYYY-MM-DD HH:MM:SS) followed by value
        # The datetime has format "2017-05-01 00:00:00" and value is a float
        match = re.match(r'^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+(-?\d+\.?\d*)$', line)
        if match:
            timestamps.append(match.group(1))
            values.append(float(match.group(2)))
        else:
            # Try alternative format: just value per line (if timestamps are implicit)
            try:
                values.append(float(line))
                timestamps.append(None)
            except ValueError:
                continue
    
    return timestamps, values


def infer_frequency(timestamps: List[pd.Timestamp]) -> Optional[pd.Timedelta]:
    """
    Infer the most common frequency from a list of timestamps.
    
    Args:
        timestamps: List of pandas Timestamp objects
        
    Returns:
        Most common time delta, or None if cannot be inferred
    """
    if len(timestamps) < 2:
        return None
    
    # Calculate all time differences
    diffs = []
    for i in range(1, len(timestamps)):
        diff = timestamps[i] - timestamps[i-1]
        if diff > pd.Timedelta(0):  # Only positive differences
            diffs.append(diff)
    
    if not diffs:
        return None
    
    # Return the most common difference (mode)
    from collections import Counter
    diff_counts = Counter(diffs)
    most_common_diff = diff_counts.most_common(1)[0][0]
    
    return most_common_diff


def parse_time_series_to_dataframe(
    data_str: str, 
    series_id: str = "series_0",
    default_freq: str = "1H"
) -> pd.DataFrame:
    """
    Parse time series string into a DataFrame suitable for predict_time_series.
    
    Handles various scenarios:
    1. All timestamps present: use them directly
    2. No timestamps: generate synthetic ones with default_freq
    3. Partial timestamps: infer frequency from available timestamps
    
    Args:
        data_str: Time series data as a string with format "timestamp value\n..."
        series_id: Identifier for the time series (default: "series_0")
        default_freq: Default frequency if no timestamps available (default: "1H" for hourly)
    
    Returns:
        DataFrame with columns: 'id', 'timestamp', 'target'
    
    Example:
        >>> df = parse_time_series_to_dataframe("2017-05-01 00:00:00 11.588\\n2017-05-01 01:00:00 10.918")
        >>> print(df.columns.tolist())
        ['id', 'timestamp', 'target']
    """
    timestamps, values = parse_time_series_string(data_str)
    
    if not values:
        raise ValueError("No valid data points found in the input string")
    
    # Check how many valid timestamps we have
    valid_timestamps = [ts for ts in timestamps if ts is not None]
    valid_ts_indices = [i for i, ts in enumerate(timestamps) if ts is not None]
    
    datetime_list = []
    
    if len(valid_timestamps) == len(timestamps) and len(valid_timestamps) > 0:
        # Case 1: All timestamps present - use them directly
        datetime_list = [pd.to_datetime(ts) for ts in timestamps]
        
    elif len(valid_timestamps) == 0:
        # Case 2: No timestamps at all - generate synthetic ones
        # Use current time as base or a reasonable default
        base_time = pd.Timestamp.now().floor('H')  # Round to current hour
        freq = pd.Timedelta(default_freq)
        datetime_list = [base_time + freq * i for i in range(len(values))]
        
    elif len(valid_timestamps) >= 2:
        # Case 3: Some timestamps available - infer frequency and fill gaps
        parsed_valid_ts = [pd.to_datetime(ts) for ts in valid_timestamps]
        inferred_freq = infer_frequency(parsed_valid_ts)
        
        if inferred_freq is None:
            inferred_freq = pd.Timedelta(default_freq)
        
        # Use the first valid timestamp as anchor
        first_valid_idx = valid_ts_indices[0]
        first_valid_ts = pd.to_datetime(valid_timestamps[0])
        
        # Generate all timestamps based on inferred frequency
        for i in range(len(values)):
            offset = i - first_valid_idx
            datetime_list.append(first_valid_ts + inferred_freq * offset)
    else:
        # Case 4: Only one timestamp - use it as anchor with default frequency
        valid_idx = valid_ts_indices[0]
        anchor_ts = pd.to_datetime(valid_timestamps[0])
        freq = pd.Timedelta(default_freq)
        
        for i in range(len(values)):
            offset = i - valid_idx
            datetime_list.append(anchor_ts + freq * offset)
    
    df = pd.DataFrame({
        'id': [series_id] * len(values),
        'timestamp': datetime_list,
        'target': values
    })
    
    return df


def format_predictions_to_string(
    pred_df: pd.DataFrame,
    last_timestamp: str = None,
    freq_hours: int = 1
) -> str:
    """
    Format prediction DataFrame into a string matching input format.
    
    Args:
        pred_df: DataFrame with predictions (from predict_time_series)
        last_timestamp: Last timestamp from input data (to continue from)
        freq_hours: Frequency in hours between data points (default: 1)
    
    Returns:
        String with format "timestamp value\n..."
    
    Example:
        >>> pred_str = format_predictions_to_string(pred_df, "2017-05-01 23:00:00")
        >>> print(pred_str[:50])
        "2017-05-02 00:00:00 12.345\n2017-05-02 01:00:00..."
    """
    lines = []
    
    # Get prediction values
    # Chronos2 returns columns like: ['id', 'timestamp', 'target_name', 'predictions', '0.5']
    # The actual predictions are in the '0.5' column (median quantile) or 'target_0.5'
    value_col = None
    
    # First, try to find the quantile column (e.g., '0.5', 'target_0.5')
    for col in pred_df.columns:
        col_str = str(col)
        if col_str == '0.5' or col_str == 'target_0.5':
            value_col = col
            break
    
    # If not found, try 'predictions' column
    if value_col is None and 'predictions' in pred_df.columns:
        value_col = 'predictions'
    
    # Fallback: use the last numeric column
    if value_col is None:
        numeric_cols = pred_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            value_col = numeric_cols[-1]
        else:
            raise ValueError("No numeric prediction column found in DataFrame")
    
    # Check if DataFrame has timestamp column
    if 'timestamp' in pred_df.columns:
        for _, row in pred_df.iterrows():
            ts = row['timestamp']
            if isinstance(ts, pd.Timestamp):
                ts_str = ts.strftime('%Y-%m-%d %H:%M:%S')
            else:
                ts_str = str(ts)
            value = row[value_col]
            # Ensure value is numeric before formatting
            try:
                value = float(value)
                lines.append(f"{ts_str} {value:.3f}")
            except (ValueError, TypeError):
                lines.append(f"{ts_str} {value}")
    else:
        # Generate timestamps from last_timestamp
        if last_timestamp:
            base_time = pd.to_datetime(last_timestamp)
        else:
            base_time = pd.Timestamp('2017-01-01')
        
        for i, (_, row) in enumerate(pred_df.iterrows()):
            ts = base_time + pd.Timedelta(hours=(i + 1) * freq_hours)
            ts_str = ts.strftime('%Y-%m-%d %H:%M:%S')
            value = row[value_col]
            # Ensure value is numeric before formatting
            try:
                value = float(value)
                lines.append(f"{ts_str} {value:.3f}")
            except (ValueError, TypeError):
                lines.append(f"{ts_str} {value}")
    
    return '\n'.join(lines)


def get_last_timestamp(data_str: str) -> Optional[str]:
    """
    Extract the last timestamp from time series string.

    Args:
        data_str: Time series data string

    Returns:
        Last timestamp as string, or None if not found
    """
    timestamps, _ = parse_time_series_string(data_str)
    if timestamps and timestamps[-1]:
        return timestamps[-1]
    return None


def _sanitize_value(value: Any) -> Any:
    """Sanitize a single value to ensure no NaN/Inf"""
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_sanitize_scalar(v) for v in value]
    return _sanitize_scalar(value)


def _sanitize_scalar(value: Any) -> Any:
    """Sanitize a scalar value"""
    if isinstance(value, str):
        return value
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(scalar):
        return 0.0
    return scalar


def extract_basic_statistics(data: List[float], seasonal_period: int = 24) -> Dict[str, Any]:
    """
    Extract basic statistical features from time series data.

    Computes 10 key features: median, MAD, ACF(1), ACF(seasonal),
    peak frequency, spectral entropy, CUSUM max, quantile kurtosis,
    mean absolute correlation, and PCA variance ratio.

    Args:
        data: List of time series values
        seasonal_period: Period for seasonal autocorrelation (default: 24)

    Returns:
        Dictionary with sanitized feature values (no NaN/Inf)
    """
    if not data or len(data) == 0:
        return {
            'median': 0.0, 'mad': 0.0, 'acf1': 0.0, 'acf_seasonal': 0.0,
            'peak_freq': 0.0, 'spec_entropy': 0.0, 'cusum_max': 0.0,
            'qkurt': 0.0, 'mean_abs_corr': 0.0, 'pca_var_ratio1': 0.0
        }

    arr = np.array(data, dtype=float).reshape(-1, 1)

    features = {}

    features['median'] = float(np.median(arr))

    median_val = np.median(arr, axis=0)
    features['mad'] = float(np.median(np.abs(arr - median_val)))

    try:
        acf_result = acf(arr[:, 0], nlags=1, fft=True)
        features['acf1'] = float(acf_result[1]) if len(acf_result) > 1 else 0.0
    except Exception:
        features['acf1'] = 0.0

    try:
        if len(arr) > seasonal_period:
            acf_result = acf(arr[:, 0], nlags=seasonal_period, fft=True)
            features['acf_seasonal'] = float(acf_result[seasonal_period])
        else:
            features['acf_seasonal'] = 0.0
    except Exception:
        features['acf_seasonal'] = 0.0

    try:
        fft_vals = fft(arr[:, 0])
        freqs = fftfreq(len(arr), d=1.0)
        power = np.abs(fft_vals[:len(fft_vals) // 2]) ** 2
        pos_freqs = freqs[:len(freqs) // 2]
        if len(power) > 0:
            peak_idx = int(np.argmax(power))
            features['peak_freq'] = float(abs(pos_freqs[peak_idx]))
            power_norm = power / np.sum(power)
            power_norm = power_norm[power_norm > 0]
            if len(power_norm) > 1:
                features['spec_entropy'] = float(-np.sum(power_norm * np.log2(power_norm)))
            else:
                features['spec_entropy'] = 0.0
        else:
            features['peak_freq'] = 0.0
            features['spec_entropy'] = 0.0
    except Exception:
        features['peak_freq'] = 0.0
        features['spec_entropy'] = 0.0

    try:
        median_val = np.median(arr[:, 0])
        deviations = arr[:, 0] - median_val
        cusum = np.cumsum(deviations)
        features['cusum_max'] = float(np.max(np.abs(cusum)))
    except Exception:
        features['cusum_max'] = 0.0

    try:
        q975 = np.percentile(arr[:, 0], 97.5)
        q025 = np.percentile(arr[:, 0], 2.5)
        q75 = np.percentile(arr[:, 0], 75)
        q25 = np.percentile(arr[:, 0], 25)
        denom = q75 - q25
        features['qkurt'] = float((q975 - q025) / denom) if denom != 0 else 0.0
    except Exception:
        features['qkurt'] = 0.0

    features['mean_abs_corr'] = 0.0

    try:
        if arr.shape[1] >= 2:
            data_std = (arr - np.mean(arr, axis=0)) / (np.std(arr, axis=0) + 1e-8)
            pca = PCA(n_components=min(arr.shape[1], arr.shape[0]))
            pca.fit(data_std)
            features['pca_var_ratio1'] = float(pca.explained_variance_ratio_[0])
        else:
            features['pca_var_ratio1'] = 1.0
    except Exception:
        features['pca_var_ratio1'] = 0.0

    return {k: _sanitize_value(v) for k, v in features.items()}


def extract_within_channel_dynamics(
    data: List[float],
    changepoint_penalty: float = 5.0,
    changepoint_max: int = 5,
    peak_prominence: float = 1.0
) -> Dict[str, Any]:
    """
    Extract within-channel dynamics features.

    Includes changepoint detection, slope analysis, and peak detection.

    Args:
        data: List of time series values
        changepoint_penalty: Penalty for changepoint detection
        changepoint_max: Maximum number of changepoints
        peak_prominence: Prominence threshold for peak detection

    Returns:
        Dictionary with sanitized feature values
    """
    if not data or len(data) == 0:
        return {
            'changepoint_count': 0.0, 'changepoint_score': 0.0,
            'slope_max': 0.0, 'slope_second_diff_max': 0.0,
            'monotone_duration': 0.0, 'peak_count': 0.0,
            'peak_max_width': 0.0, 'peak_spacing_cv': 0.0
        }

    arr = np.array(data, dtype=float)
    features = {}

    if len(arr) < 10 or rpt is None:
        features['changepoint_count'] = 0.0
        features['changepoint_score'] = 0.0
    else:
        try:
            algo = rpt.Pelt(model='rbf').fit(arr)
            bkps = algo.predict(pen=changepoint_penalty)
            count = max(0, len(bkps) - 1)
            features['changepoint_count'] = float(min(count, changepoint_max))
            if count > 0:
                segments = np.split(arr, bkps[:-1])
                seg_means = [np.mean(seg) for seg in segments if len(seg) > 0]
                if len(seg_means) > 1:
                    diff = np.abs(np.diff(seg_means))
                    features['changepoint_score'] = float(np.max(diff))
                else:
                    features['changepoint_score'] = 0.0
            else:
                features['changepoint_score'] = 0.0
        except Exception:
            features['changepoint_count'] = 0.0
            features['changepoint_score'] = 0.0

    if len(arr) < 3:
        features['slope_max'] = 0.0
        features['slope_second_diff_max'] = 0.0
        features['monotone_duration'] = 0.0
    else:
        diff1 = np.diff(arr)
        diff2 = np.diff(arr, n=2)
        features['slope_max'] = float(np.max(np.abs(diff1)))
        features['slope_second_diff_max'] = float(np.max(np.abs(diff2))) if len(diff2) else 0.0

        if len(diff1) == 0:
            features['monotone_duration'] = 0.0
        else:
            signs = np.sign(diff1)
            longest = current = 1
            prev = signs[0]
            for s in signs[1:]:
                if s == 0 or s == prev:
                    current += 1
                else:
                    longest = max(longest, current)
                    current = 1
                if s != 0:
                    prev = s
            longest = max(longest, current)
            features['monotone_duration'] = float(longest / max(len(arr), 1))

    if len(arr) < 3:
        features['peak_count'] = 0.0
        features['peak_max_width'] = 0.0
        features['peak_spacing_cv'] = 0.0
    else:
        try:
            normalized = (arr - np.mean(arr)) / (np.std(arr) + 1e-8)
            peaks, _ = find_peaks(normalized, prominence=peak_prominence)
            features['peak_count'] = float(len(peaks))
            if len(peaks) == 0:
                features['peak_max_width'] = 0.0
                features['peak_spacing_cv'] = 0.0
            else:
                widths = peak_widths(normalized, peaks)[0]
                features['peak_max_width'] = float(np.max(widths) if len(widths) else 0.0)
                if len(peaks) > 1:
                    spacing = np.diff(peaks)
                    features['peak_spacing_cv'] = float(np.std(spacing) / (np.mean(spacing) + 1e-8))
                else:
                    features['peak_spacing_cv'] = 0.0
        except Exception:
            features['peak_count'] = 0.0
            features['peak_max_width'] = 0.0
            features['peak_spacing_cv'] = 0.0

    return {k: _sanitize_value(v) for k, v in features.items()}


def extract_forecast_residuals(data: List[float], ar_order: int = 1) -> Dict[str, Any]:
    """
    Extract forecast residual features using AutoReg model.

    Args:
        data: List of time series values
        ar_order: Order of autoregressive model

    Returns:
        Dictionary with residual features
    """
    if not data or len(data) <= ar_order + 2:
        return {
            'residual_mean': 0.0, 'residual_max': 0.0,
            'residual_exceed_ratio': 0.0, 'residual_acf1': 0.0,
            'residual_concentration': 0.0
        }

    arr = np.array(data, dtype=float)
    features = {}

    try:
        model = AutoReg(arr, lags=ar_order, old_names=False).fit()
        resid = model.resid
    except Exception:
        resid = arr[ar_order:] - arr[:-ar_order]

    if len(resid) == 0:
        resid = np.zeros(1)

    features['residual_mean'] = float(np.mean(resid))
    features['residual_max'] = float(np.max(np.abs(resid)))

    std_val = np.std(resid) + 1e-8
    features['residual_exceed_ratio'] = float(np.mean(np.abs(resid) > 2 * std_val))

    try:
        acf_val = acf(resid, nlags=1, fft=False)
        features['residual_acf1'] = float(acf_val[1]) if len(acf_val) > 1 else 0.0
    except Exception:
        features['residual_acf1'] = 0.0

    sorted_resid = np.sort(np.abs(resid))
    tail_start = int(0.9 * len(sorted_resid))
    tail_sum = np.sum(sorted_resid[tail_start:])
    total_sum = np.sum(sorted_resid) + 1e-8
    features['residual_concentration'] = float(tail_sum / total_sum)

    return {k: _sanitize_value(v) for k, v in features.items()}


def extract_data_quality(data: List[float], quantization_bins: int = 25, flatline_tol: float = 1e-3) -> Dict[str, Any]:
    """
    Extract data quality features.

    Args:
        data: List of time series values
        quantization_bins: Number of bins for quantization analysis
        flatline_tol: Tolerance for flatline detection

    Returns:
        Dictionary with quality features
    """
    if not data or len(data) == 0:
        return {
            'quality_quantization_score': 0.0,
            'quality_saturation_ratio': 0.0,
            'quality_constant_channel_ratio': 0.0,
            'quality_dropout_ratio': 0.0
        }

    arr = np.array(data, dtype=float).reshape(-1, 1)
    flattened = arr.flatten()
    features = {}

    try:
        hist, _ = np.histogram(flattened, bins=quantization_bins)
        features['quality_quantization_score'] = float(np.max(hist) / len(flattened))
    except Exception:
        features['quality_quantization_score'] = 0.0

    try:
        span = np.max(flattened) - np.min(flattened)
        margin = max(span * 0.01, flatline_tol)
        saturation = np.logical_or(
            flattened <= np.min(flattened) + margin,
            flattened >= np.max(flattened) - margin
        )
        features['quality_saturation_ratio'] = float(np.mean(saturation))
    except Exception:
        features['quality_saturation_ratio'] = 0.0

    try:
        features['quality_constant_channel_ratio'] = float(np.mean(np.std(arr, axis=0) < flatline_tol))
    except Exception:
        features['quality_constant_channel_ratio'] = 0.0

    try:
        features['quality_dropout_ratio'] = float(np.mean(~np.isfinite(flattened)))
    except Exception:
        features['quality_dropout_ratio'] = 0.0

    return {k: _sanitize_value(v) for k, v in features.items()}


def extract_event_summary(data: List[float]) -> Dict[str, Any]:
    """
    Extract event summary features by segmenting the series.

    Args:
        data: List of time series values

    Returns:
        Dictionary with event summary features
    """
    if not data or len(data) == 0:
        return {
            'event_segment_count': 0.0,
            'event_counts': [0.0, 0.0, 0.0, 0.0],
            'event_dominant_pattern': 0.0
        }

    arr = np.array(data, dtype=float)

    if len(arr) < 4:
        segments = [arr]
    else:
        diffs = np.diff(arr)
        threshold = np.std(diffs) * 1.5 + 1e-6
        change_points = [0]
        for idx in range(1, len(diffs)):
            if abs(diffs[idx] - diffs[idx - 1]) > threshold:
                change_points.append(idx)
        change_points.append(len(arr) - 1)
        segments = []
        for start, end in zip(change_points[:-1], change_points[1:]):
            seg = arr[start:end + 1]
            if len(seg) > 0:
                segments.append(seg)
        segments = segments[:8]
        if len(segments) < 3 and len(arr) >= 3:
            split_size = max(1, len(arr) // 3)
            segments = [arr[:split_size], arr[split_size:2 * split_size], arr[2 * split_size:]]

    slope_threshold = np.std(arr) * 0.05 + 1e-6
    event_counts = Counter({'rise': 0, 'fall': 0, 'flat': 0, 'oscillation': 0})

    for segment in segments:
        if len(segment) < 2:
            continue
        slope = segment[-1] - segment[0]
        segment_std = np.std(segment)
        if slope > slope_threshold:
            event_counts['rise'] += 1
        elif slope < -slope_threshold:
            event_counts['fall'] += 1
        elif segment_std > slope_threshold * 2:
            event_counts['oscillation'] += 1
        else:
            event_counts['flat'] += 1

    order = ['rise', 'fall', 'flat', 'oscillation']
    counts_list = [float(event_counts[name]) for name in order]
    dominant_idx = int(np.argmax(counts_list)) if counts_list else 0

    features = {
        'event_segment_count': float(len(segments)),
        'event_counts': counts_list,
        'event_dominant_pattern': float(dominant_idx)
    }

    return {k: _sanitize_value(v) for k, v in features.items()}


def format_basic_statistics(features: Dict[str, Any]) -> str:
    """
    Format basic statistics features for human-readable output.

    Args:
        features: Dictionary from extract_basic_statistics

    Returns:
        Formatted string (5-10 lines)
    """
    lines = []
    lines.append("Basic Statistics:")
    lines.append(f"  Median: {features.get('median', 0.0):.4f}")
    lines.append(f"  MAD: {features.get('mad', 0.0):.4f}")
    lines.append(f"  ACF(1): {features.get('acf1', 0.0):.4f}")
    lines.append(f"  ACF(seasonal): {features.get('acf_seasonal', 0.0):.4f}")
    lines.append(f"  Peak Frequency: {features.get('peak_freq', 0.0):.4f}")
    lines.append(f"  Spectral Entropy: {features.get('spec_entropy', 0.0):.4f}")
    lines.append(f"  CUSUM Max: {features.get('cusum_max', 0.0):.4f}")
    lines.append(f"  Quantile Kurtosis: {features.get('qkurt', 0.0):.4f}")
    return "\n".join(lines)


def format_within_channel_dynamics(features: Dict[str, Any]) -> str:
    """
    Format within-channel dynamics features for human-readable output.

    Args:
        features: Dictionary from extract_within_channel_dynamics

    Returns:
        Formatted string (5-10 lines)
    """
    lines = []
    lines.append("Within-Channel Dynamics:")
    lines.append(f"  Changepoint Count: {features.get('changepoint_count', 0.0):.1f}")
    lines.append(f"  Changepoint Score: {features.get('changepoint_score', 0.0):.4f}")
    lines.append(f"  Max Slope: {features.get('slope_max', 0.0):.4f}")
    lines.append(f"  Max Second Diff: {features.get('slope_second_diff_max', 0.0):.4f}")
    lines.append(f"  Monotone Duration: {features.get('monotone_duration', 0.0):.4f}")
    lines.append(f"  Peak Count: {features.get('peak_count', 0.0):.1f}")
    lines.append(f"  Peak Max Width: {features.get('peak_max_width', 0.0):.4f}")
    lines.append(f"  Peak Spacing CV: {features.get('peak_spacing_cv', 0.0):.4f}")
    return "\n".join(lines)


def format_forecast_residuals(features: Dict[str, Any]) -> str:
    """
    Format forecast residual features for human-readable output.

    Args:
        features: Dictionary from extract_forecast_residuals

    Returns:
        Formatted string (5-10 lines)
    """
    lines = []
    lines.append("Forecast Residuals:")
    lines.append(f"  Residual Mean: {features.get('residual_mean', 0.0):.4f}")
    lines.append(f"  Residual Max: {features.get('residual_max', 0.0):.4f}")
    lines.append(f"  Exceed Ratio: {features.get('residual_exceed_ratio', 0.0):.4f}")
    lines.append(f"  ACF(1): {features.get('residual_acf1', 0.0):.4f}")
    lines.append(f"  Concentration: {features.get('residual_concentration', 0.0):.4f}")
    return "\n".join(lines)


def format_data_quality(features: Dict[str, Any]) -> str:
    """
    Format data quality features for human-readable output.

    Args:
        features: Dictionary from extract_data_quality

    Returns:
        Formatted string (5-10 lines)
    """
    lines = []
    lines.append("Data Quality:")
    lines.append(f"  Quantization Score: {features.get('quality_quantization_score', 0.0):.4f}")
    lines.append(f"  Saturation Ratio: {features.get('quality_saturation_ratio', 0.0):.4f}")
    lines.append(f"  Constant Channel Ratio: {features.get('quality_constant_channel_ratio', 0.0):.4f}")
    lines.append(f"  Dropout Ratio: {features.get('quality_dropout_ratio', 0.0):.4f}")
    return "\n".join(lines)


def format_event_summary(features: Dict[str, Any]) -> str:
    """
    Format event summary features for human-readable output.

    Args:
        features: Dictionary from extract_event_summary

    Returns:
        Formatted string (5-10 lines)
    """
    lines = []
    lines.append("Event Summary:")
    lines.append(f"  Segment Count: {features.get('event_segment_count', 0.0):.1f}")
    counts = features.get('event_counts', [0.0, 0.0, 0.0, 0.0])
    lines.append(f"  Rise Events: {counts[0]:.1f}")
    lines.append(f"  Fall Events: {counts[1]:.1f}")
    lines.append(f"  Flat Events: {counts[2]:.1f}")
    lines.append(f"  Oscillation Events: {counts[3]:.1f}")
    pattern_names = ['rise', 'fall', 'flat', 'oscillation']
    dominant_idx = int(features.get('event_dominant_pattern', 0))
    lines.append(f"  Dominant Pattern: {pattern_names[dominant_idx]}")
    return "\n".join(lines)
