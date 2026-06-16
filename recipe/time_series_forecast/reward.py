import string
import re
import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Tuple


RAW_BASELINE_METRICS: dict[str, tuple[float, float]] = {
    "ETTH1": (2.639, 1.157),
    "ETTH2": (32.922, 4.606),
    "ETTM1": (0.381, 0.486),
    "ETTM2": (28.272, 3.705),
    "WIND": (305.786, 12.031),
    "NP": (19.932, 2.34),
    "PJM": (20.642, 3.255),
    "BE": (134.886, 8.414),
    "DE": (175.534, 11.571),
    "FR": (39.926, 4.878),
}


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


def decompose(x: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Decompose time series into seasonal and trend components."""
    x = np.array(x)
    x = torch.from_numpy(x).float()
    x = x.unsqueeze(0).unsqueeze(2)
    
    kernel_size = min(25, len(x.squeeze()) // 2)
    if kernel_size < 3:
        kernel_size = 3
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    model = series_decomp(kernel_size=kernel_size)
    season, trend = model(x)

    season = season.squeeze(0).squeeze(1).detach().numpy()
    trend = trend.squeeze(0).squeeze(1).detach().numpy()
    return season, trend


def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    """Calculate mean squared error."""
    return float(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))


def _normalize_dataset_name(dataset_name: Optional[str]) -> Optional[str]:
    if dataset_name is None:
        return None
    normalized = str(dataset_name).strip().upper()
    if normalized == "WIND":
        return "WIND"
    return normalized or None


def compute_raw_mse_mae(solution_str: str, ground_truth: str) -> tuple[float, float]:
    """Compute raw-scale MSE/MAE using the full ground-truth horizon as the gate."""
    gt_values = extract_ground_truth_values(ground_truth)
    pred_values = extract_values_from_time_series_string(solution_str)

    if not gt_values or len(pred_values) < len(gt_values):
        return float("nan"), float("nan")

    gt_arr = np.array(gt_values, dtype=float)
    pred_arr = np.array(pred_values[: len(gt_values)], dtype=float)
    if not np.all(np.isfinite(gt_arr)) or not np.all(np.isfinite(pred_arr)):
        return float("nan"), float("nan")

    diff = pred_arr - gt_arr
    return float(np.mean(diff**2)), float(np.mean(np.abs(diff)))


def _metric_margin(metric: float, target: float) -> float:
    if not np.isfinite(metric) or not np.isfinite(target) or target <= 0:
        return 0.0
    return float((target - metric) / target)


def _same_prediction_values(left: str, right: str) -> bool:
    left_values = extract_values_from_time_series_string(left)
    right_values = extract_values_from_time_series_string(right)
    if not left_values or len(left_values) != len(right_values):
        return False
    left_arr = np.array(left_values, dtype=float)
    right_arr = np.array(right_values, dtype=float)
    if not np.all(np.isfinite(left_arr)) or not np.all(np.isfinite(right_arr)):
        return False
    return bool(np.allclose(left_arr, right_arr, rtol=1e-6, atol=1e-6))


def compute_raw_relative_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
) -> float:
    """
    Reward raw-scale improvement toward the published baseline and tool forecast.

    The old normalized/shape reward can give high scores to copied tool forecasts
    whose raw MSE/MAE are still worse than the baseline. This score uses raw MSE/MAE
    as the main objective and makes exact tool copies non-positive.
    """
    if solution_str is None:
        return -1.0

    format_score = compute_format_score(solution_str)
    if format_score < 0:
        return format_score

    mse, mae = compute_raw_mse_mae(solution_str, ground_truth)
    if not np.isfinite(mse) or not np.isfinite(mae):
        return -0.5

    extra_info = extra_info or {}
    dataset_name = _normalize_dataset_name(extra_info.get("dataset_name") or data_source)
    targets: list[tuple[float, float]] = []

    baseline = RAW_BASELINE_METRICS.get(dataset_name or "")
    if baseline:
        targets.append(baseline)

    reference_prediction = extra_info.get("reference_prediction")
    ref_mse = ref_mae = float("nan")
    if reference_prediction:
        ref_mse, ref_mae = compute_raw_mse_mae(str(reference_prediction), ground_truth)
        if np.isfinite(ref_mse) and np.isfinite(ref_mae):
            targets.append((ref_mse, ref_mae))

    if targets:
        target_mse = min(target[0] for target in targets)
        target_mae = min(target[1] for target in targets)
        score = 0.5 * _metric_margin(mse, target_mse) + 0.5 * _metric_margin(mae, target_mae)
    else:
        # Unknown datasets still get a monotonic raw-MSE signal, but no positive
        # baseline margin is invented.
        score = compute_mse_score(solution_str, ground_truth) - 0.5

    if reference_prediction and np.isfinite(ref_mse) and np.isfinite(ref_mae):
        delta = 0.5 * _metric_margin(mse, ref_mse) + 0.5 * _metric_margin(mae, ref_mae)
        score += 0.5 * float(np.clip(delta, -1.0, 1.0))
        if _same_prediction_values(solution_str, str(reference_prediction)):
            if np.isclose(ref_mse, 0.0) and np.isclose(ref_mae, 0.0):
                score = max(score, 0.0)
            else:
                # Exact reference copies are only acceptable when the reference is already perfect.
                # Otherwise push the policy away from the "wrap tool output in <answer>" local optimum.
                ref_error_scale = float(np.clip((ref_mse + ref_mae) / (ref_mse + ref_mae + 1.0), 0.0, 1.0))
                copy_penalty = 0.1 + 0.2 * ref_error_scale
                score = min(score, -copy_penalty)

    return float(np.clip(score, -1.0, 1.0))


def mean_squared_error_season_trend(y_true: List[float], y_pred: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """Calculate MSE for seasonal and trend components separately.
    
    Returns:
        Tuple of (mse_season, mse_trend), or (None, None) if decomposition fails.
    """
    if len(y_true) < 5 or len(y_pred) < 5:
        # Not enough data for decomposition
        return None, None
    
    try:
        season_true, trend_true = decompose(y_true)
        season_pred, trend_pred = decompose(y_pred)
        mse_season = float(np.mean((season_true - season_pred) ** 2))
        mse_trend = float(np.mean((trend_true - trend_pred) ** 2))
        return mse_season, mse_trend
    except Exception:
        return None, None


def extract_answer(text: str) -> str:
    """Extract content within <answer> tags."""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    return match.group(1).strip() if match else text


def extract_values_from_time_series_string(text: str) -> List[float]:
    """
    Extract numeric values from time series string format.
    
    Expected formats:
    1. "2017-05-01 00:00:00 11.588" (timestamp + value)
    2. "| 1 | 11.588 |" (table format)
    3. Just numbers on each line
    
    Args:
        text: Text containing time series data
        
    Returns:
        List of float values
    """
    text = extract_answer(text)
    values = []
    
    lines = text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Pattern 1: timestamp format "2017-05-01 00:00:00 11.588"
        match = re.search(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+(-?\d+\.?\d*)', line)
        if match:
            try:
                values.append(float(match.group(1)))
                continue
            except ValueError:
                pass
        
        # Pattern 2: table format "| 1 | 11.588 |"
        match = re.search(r'\|\s*\d+\s*\|\s*(-?\d+\.?\d*)\s*\|', line)
        if match:
            try:
                values.append(float(match.group(1)))
                continue
            except ValueError:
                pass
        
        # Pattern 3: just the last number on the line
        match = re.search(r'(-?\d+\.?\d*)$', line)
        if match:
            try:
                values.append(float(match.group(1)))
                continue
            except ValueError:
                pass
    
    return values


def extract_ground_truth_values(text: str) -> List[float]:
    """Extract ground truth values from time series string."""
    return extract_values_from_time_series_string(text)


def normalize_for_reward(x_list: List[float], y_list: List[float]) -> Tuple[List[float], List[float]]:
    """
    Normalize both prediction and ground truth for fair comparison.
    
    Uses GROUND TRUTH's mean and std as normalization parameters.
    This ensures stable normalization regardless of prediction quality.
    """
    x_array = np.array(x_list)
    y_array = np.array(y_list)
    
    # Use ground truth's statistics for stable normalization
    mu = np.nanmean(y_array)
    std = np.nanstd(y_array)
    std = max(std, 1e-8)  # Avoid division by zero
    
    x_result = (x_array - mu) / std
    y_result = (y_array - mu) / std
    
    return x_result.tolist(), y_result.tolist()


def compute_format_score(solution_str: str) -> float:
    """
    Check if the solution follows the required format with <answer> tags.
    
    Returns:
        0.0 if format is correct, -1.0 otherwise
    """
    if solution_str is None:
        return -1.0
    
    try:
        # Check for <answer>...</answer> pattern
        answer_match = re.search(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)
        
        if answer_match:
            return 0.0
        return -1.0
    except Exception as e:
        print(f"[DEBUG] Error in compute_format_score: {e}")
        return -1.0


def compute_length_score(solution_str: str, ground_truth: str) -> float:
    """
    Reward for generating the correct number of predictions.
    
    Returns:
        0.1 if prediction length >= ground truth length
        Proportional score otherwise
    """
    gt_values = extract_ground_truth_values(ground_truth)
    pred_values = extract_values_from_time_series_string(solution_str)
    
    if not gt_values:
        return 0.0
    
    if len(pred_values) >= len(gt_values):
        return 0.1
    else:
        return 0.1 * len(pred_values) / len(gt_values)


def compute_mse_score(solution_str: str, ground_truth: str) -> float:
    """
    Compute score based on MSE between prediction and ground truth.
    
    Uses log transformation: score = 1 / (1 + log(1 + mse))
    
    This design ensures:
    1. MSE=0 -> score=1 (perfect prediction)
    2. MSE=1 -> score=0.59
    3. MSE=10 -> score=0.29
    4. MSE=100 -> score=0.18
    5. MSE=1000 -> score=0.13
    
    Key benefits for GRPO training:
    - Log compression: even very large MSE still produces meaningful scores
    - Always differentiable: smooth gradient for optimization
    - Maintains ranking: lower MSE always gets higher score
    - Good discrimination: even poor predictions have distinguishable rewards
    
    If prediction is shorter than ground truth:
    - Calculate MSE on available predictions
    - Apply length penalty: score * (pred_len / gt_len)
    
    Returns:
        Score in range [0, 0.6]
    """
    gt_values = extract_ground_truth_values(ground_truth)
    pred_values = extract_values_from_time_series_string(solution_str)
    
    if not gt_values or not pred_values:
        return 0.0
    
    # Calculate length ratio for penalty
    pred_len = len(pred_values)
    gt_len = len(gt_values)
    
    # Use minimum length for comparison
    min_len = min(pred_len, gt_len)
    if min_len == 0:
        return 0.0
    
    # Normalize using ground truth's statistics (stable normalization)
    norm_pred, norm_gt = normalize_for_reward(pred_values[:min_len], gt_values[:min_len])
    
    # Calculate MSE on normalized values
    mse = mean_squared_error(norm_gt, norm_pred)
    
    # Transform MSE to score using log compression
    # score = 1 / (1 + log(1 + mse))
    # - Log compresses large MSE values, maintaining discrimination even for poor predictions
    # - Always positive, bounded in (0, 1]
    # - Monotonically decreasing with MSE
    score = 1.0 / (1.0 + np.log1p(mse))
    
    # Apply length penalty if prediction is shorter than ground truth
    length_ratio = min(pred_len / gt_len, 1.0)
    score = score * length_ratio
    
    return score * 0.6


def compute_season_trend_score(solution_str: str, ground_truth: str) -> float:
    """
    Compute score based on seasonal and trend decomposition.
    
    Uses log compression (consistent with compute_mse_score).
    
    Returns:
        Score up to 0.2 (0.05 for season + 0.15 for trend)
        Returns 0.0 if decomposition fails (instead of giving undeserved points)
    """
    gt_values = extract_ground_truth_values(ground_truth)
    pred_values = extract_values_from_time_series_string(solution_str)
    
    if not gt_values or not pred_values:
        return 0.0
    
    if len(gt_values) < 5:
        return 0.0
    
    # Use minimum length for comparison
    min_len = min(len(pred_values), len(gt_values))
    if min_len < 5:
        return 0.0
    
    # Normalize using ground truth's statistics
    norm_pred, norm_gt = normalize_for_reward(pred_values[:min_len], gt_values[:min_len])
    
    # Calculate MSE for seasonal and trend components
    mse_season, mse_trend = mean_squared_error_season_trend(norm_gt, norm_pred)
    
    # If decomposition failed, return 0 (not undeserved points)
    if mse_season is None or mse_trend is None:
        return 0.0
    
    # Transform to scores using log compression (consistent with compute_mse_score)
    score_season = 1.0 / (1.0 + np.log1p(mse_season))
    score_trend = 1.0 / (1.0 + np.log1p(mse_trend))
    
    # Apply length penalty if prediction is shorter
    length_ratio = min(len(pred_values) / len(gt_values), 1.0)
    
    return (0.05 * score_season + 0.15 * score_trend) * length_ratio


def find_change_points(data: List[float]) -> Tuple[List[int], List[int]]:
    """
    Find local maxima and minima (change points) in the data.
    
    Returns:
        Tuple of (max_indices, min_indices)
    """
    if len(data) < 5:
        return [], []
    
    change_point_max = []
    change_point_min = []
    
    # Mirror padding
    data_mirror_forward = [data[2], data[1]]
    data_mirror_backward = [data[-2], data[-1]]
    data_new = data_mirror_forward + list(data) + data_mirror_backward
    
    for i in range(len(data)):
        # Check for local maximum
        if (data_new[i+2] >= data_new[i] and 
            data_new[i+2] >= data_new[i+1] and 
            data_new[i+2] >= data_new[i+3] and 
            data_new[i+2] >= data_new[i+4]):
            change_point_max.append(i)
        
        # Check for local minimum
        if (data_new[i+2] <= data_new[i] and 
            data_new[i+2] <= data_new[i+1] and 
            data_new[i+2] <= data_new[i+3] and 
            data_new[i+2] <= data_new[i+4]):
            change_point_min.append(i)
    
    return change_point_max, change_point_min


def compute_change_point_score(solution_str: str, ground_truth: str, tolerance: int = 2) -> float:
    """
    Compute score based on correctly predicted change points (local max/min).
    
    Args:
        solution_str: Model's prediction string
        ground_truth: Ground truth string
        tolerance: Allow change point to be off by this many time steps (default: 2)
    
    Returns:
        Score up to 0.2 (0.1 for max + 0.1 for min)
    """
    gt_values = extract_ground_truth_values(ground_truth)
    pred_values = extract_values_from_time_series_string(solution_str)
    
    if not gt_values or not pred_values:
        return 0.0
    
    # Use minimum length for comparison
    min_len = min(len(pred_values), len(gt_values))
    if min_len < 5:
        return 0.0
    
    gt_max, gt_min = find_change_points(gt_values[:min_len])
    pred_max, pred_min = find_change_points(pred_values[:min_len])
    
    def count_hits_with_tolerance(pred_points: List[int], gt_points: List[int], tol: int) -> int:
        """Count predicted points that are within tolerance of a ground truth point."""
        hits = 0
        used_gt = set()
        for p in pred_points:
            for g in gt_points:
                if g not in used_gt and abs(p - g) <= tol:
                    hits += 1
                    used_gt.add(g)
                    break
        return hits
    
    # Count correct predictions with tolerance
    max_hits = count_hits_with_tolerance(pred_max, gt_max, tolerance)
    min_hits = count_hits_with_tolerance(pred_min, gt_min, tolerance)
    
    # Calculate scores
    max_score = max_hits / len(gt_max) * 0.1 if gt_max else 0.0
    min_score = min_hits / len(gt_min) * 0.1 if gt_min else 0.0
    
    # Apply length penalty if prediction is shorter
    length_ratio = min(len(pred_values) / len(gt_values), 1.0)
    
    return (max_score + min_score) * length_ratio


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None
) -> float:
    """
    Compute the total reward score for time series prediction.
    
    The score is composed of:
    1. Format score: -1.0 to 0.0 (penalty for wrong format)
    2. Length score: 0.0 to 0.1 (reward for correct output length)
    3. MSE score: 0.0 to 0.6 (main prediction accuracy)
    4. Change point score: 0.0 to 0.2 (correct local extrema) [DISABLED for ablation]
    5. Season/Trend score: 0.0 to 0.2 (pattern matching) [DISABLED for ablation]
    
    Total possible range (with ablation): -1.0 to 0.7
    
    Args:
        data_source: Source identifier (unused but kept for compatibility)
        solution_str: Model's solution string with <answer> tags
        ground_truth: Ground truth time series string
        extra_info: Optional additional information (unused)
        
    Returns:
        Total reward score
    """
    if solution_str is None:
        return -1.0
    
    score = 0.0
    
    # 1. Format score
    format_score = compute_format_score(solution_str)
    score += format_score
    
    # If format is wrong, return early with penalty
    if format_score < 0:
        return score
    
    # 2. Length score
    score += compute_length_score(solution_str, ground_truth)
    
    # 3. MSE score (main accuracy metric)
    score += compute_mse_score(solution_str, ground_truth)
    
    # 4. Change point score [DISABLED for ablation experiment]
    try:
        score += compute_change_point_score(solution_str, ground_truth)
    except Exception as e:
        print(f"[DEBUG] Error in compute_change_point_score: {e}")
    
    # 5. Season/Trend decomposition score [DISABLED for ablation experiment]
    try:
        score += compute_season_trend_score(solution_str, ground_truth)
    except Exception as e:
        print(f"[DEBUG] Error in compute_season_trend_score: {e}")
    
    return score
