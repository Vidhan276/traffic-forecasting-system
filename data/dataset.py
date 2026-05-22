"""
dataset.py — PyTorch Dataset + metrics for traffic forecasting.

Provides:
  - TrafficDataset    : sliding-window (input, target) pairs for the T-GCN model.
  - create_datasets   : time-based train/val/test split + z-score normalisation.
  - compute_metrics   : MAE, RMSE, Robust-MAPE, R² with JSON-serialisable output.
  - save_metrics_json : write metrics dict to a JSON file for easy comparison.
  - load_scaler / inverse_transform : helpers for un-normalising predictions.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class TrafficDataset(Dataset):
    """
    Sliding-window dataset for traffic time-series.

    Given data of shape (T, num_nodes):
      - Input  ``x``: seq_len consecutive timesteps → (seq_len, num_nodes, 1)
      - Target ``y``: pred_len timesteps right after → (num_nodes, pred_len)

    Args:
        data    : numpy array (T, num_nodes), already normalised.
        seq_len : number of input timesteps (e.g. 12 = 1 hour @ 5-min intervals).
        pred_len: number of target timesteps (e.g. 3 = 15 min ahead).
    """

    def __init__(self, data: np.ndarray, seq_len: int = 12, pred_len: int = 3):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_samples = len(data) - seq_len - pred_len + 1

        if self.num_samples <= 0:
            raise ValueError(
                f"Not enough data: {len(data)} timesteps for "
                f"seq_len={seq_len} + pred_len={pred_len}"
            )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns one (input, target) pair.

        Returns:
            x_seq : (seq_len, num_nodes, 1) — past traffic values.
            y     : (num_nodes, pred_len)   — future values to predict.
        """
        x = self.data[idx : idx + self.seq_len]                             # (seq_len, N)
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]  # (pred_len, N)

        x_seq = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)  # (seq_len, N, 1)
        y     = torch.tensor(y.T, dtype=torch.float32)               # (N, pred_len)

        return x_seq, y


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset Creation
# ═══════════════════════════════════════════════════════════════════════════════

def create_datasets(
    traffic_data: np.ndarray,
    seq_len: int = 12,
    pred_len: int = 3,
    scaler_path: str = "ml/scaler.pkl",
) -> tuple[TrafficDataset, TrafficDataset, TrafficDataset, dict]:
    """
    Split traffic data by time into train/val/test and apply z-score normalisation.

    The split is time-based (never shuffled) to prevent future information leakage:
      - Train: first 70% of timesteps.
      - Val:   next 15%.
      - Test:  final 15%.

    Normalisation stats are computed ONLY from the training set.

    Args:
        traffic_data: numpy array (T, num_nodes).
        seq_len     : input window length.
        pred_len    : forecast horizon.
        scaler_path : where to save the scaler pickle.

    Returns:
        (train_dataset, val_dataset, test_dataset, scaler_dict)
    """
    T = len(traffic_data)
    train_end = int(T * 0.70)
    val_end   = int(T * 0.85)

    train_data = traffic_data[:train_end]
    val_data   = traffic_data[train_end:val_end]
    test_data  = traffic_data[val_end:]

    print(f"Data split (by time):")
    print(f"  Train: {len(train_data)} steps (0-{train_end - 1})")
    print(f"  Val:   {len(val_data)} steps ({train_end}-{val_end - 1})")
    print(f"  Test:  {len(test_data)} steps ({val_end}-{T - 1})")

    # z-score normalisation using training stats only
    train_mean = float(train_data.mean())
    train_std  = float(train_data.std())
    if train_std < 1e-8:
        train_std = 1.0

    print(f"Scaler: mean={train_mean:.4f}, std={train_std:.4f}")

    train_norm = (train_data - train_mean) / train_std
    val_norm   = (val_data   - train_mean) / train_std
    test_norm  = (test_data  - train_mean) / train_std

    scaler = {"mean": train_mean, "std": train_std}
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved -> {scaler_path}")

    train_dataset = TrafficDataset(train_norm, seq_len, pred_len)
    val_dataset   = TrafficDataset(val_norm,   seq_len, pred_len)
    test_dataset  = TrafficDataset(test_norm,  seq_len, pred_len)

    print(
        f"Dataset sizes: train={len(train_dataset)}, "
        f"val={len(val_dataset)}, test={len(test_dataset)}"
    )

    return train_dataset, val_dataset, test_dataset, scaler


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    mape_eps: float = 0.1,
) -> dict[str, float]:
    """
    Compute MAE, RMSE, Robust-MAPE, and R² between predictions and targets.

    Robust-MAPE fixes the original high MAPE caused by near-zero night-time
    traffic values.  Instead of dividing by |y|, we divide by max(|y|, eps),
    where eps=0.1 clips the denominator so that only truly meaningful predictions
    (where actual traffic is non-trivial) are counted.

    Args:
        predictions : predicted values, any shape that matches targets.
        targets     : ground-truth values, same shape as predictions.
        mape_eps    : minimum denominator for MAPE to avoid division by ~0.
                      Set to 0.1 so night-time values (< 0.1) are excluded.

    Returns:
        Dict with keys: "mae", "rmse", "robust_mape", "r2".
        All values are plain Python floats (JSON-serialisable).
    """
    preds   = np.asarray(predictions, dtype=np.float64).ravel()
    targets = np.asarray(targets, dtype=np.float64).ravel()

    diff = preds - targets

    # MAE
    mae = float(np.mean(np.abs(diff)))

    # RMSE
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    # Robust MAPE: clip denominator to [eps, +inf)
    denominator = np.maximum(np.abs(targets), mape_eps)
    robust_mape = float(np.mean(np.abs(diff) / denominator) * 100)

    # R² (coefficient of determination)
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((targets - targets.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    return {
        "mae":         mae,
        "rmse":        rmse,
        "robust_mape": robust_mape,
        "r2":          r2,
    }


def save_metrics_json(metrics: dict[str, Any], path: str | Path) -> None:
    """
    Save a metrics dict to a JSON file for easy comparison across runs.

    Args:
        metrics : dict of metric_name → value (must be JSON-serialisable).
        path    : output file path.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved -> {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Scaler Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def load_scaler(scaler_path: str = "ml/scaler.pkl") -> dict[str, float]:
    """Load saved z-score normalisation parameters."""
    with open(scaler_path, "rb") as f:
        return pickle.load(f)


def inverse_transform(
    data: np.ndarray | torch.Tensor,
    scaler: dict[str, float],
) -> np.ndarray | torch.Tensor:
    """
    Undo z-score normalisation: original = data * std + mean.

    Works for both numpy arrays and PyTorch tensors.
    """
    return data * scaler["std"] + scaler["mean"]
