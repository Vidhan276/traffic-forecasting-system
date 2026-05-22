"""
dataset.py — PyTorch Dataset for sliding-window traffic sequences.

This module provides:
  1. TrafficDataset  — creates (input, target) pairs for the T-GCN model
  2. create_datasets — splits data by time into train/val/test and normalizes
  3. Scaler save/load — so we can un-normalize predictions later

The sliding window approach:
  Given traffic data of shape (T, N) where T=timesteps, N=nodes:
  - Input:  seq_len consecutive timesteps   → shape (seq_len, N, 1)
  - Target: pred_len timesteps right after   → shape (N, pred_len)

Run from project root:  imported by train_model.py and evaluate.py
"""

import numpy as np
import pickle
import torch
from torch.utils.data import Dataset


class TrafficDataset(Dataset):
    """
    Sliding-window dataset for traffic time series.

    Args:
        data     : numpy array of shape (T, num_nodes), already normalized
        seq_len  : number of input timesteps  (e.g. 12 = 1 hour at 5-min intervals)
        pred_len : number of target timesteps (e.g. 3 = 15 minutes ahead)
    """

    def __init__(self, data, seq_len=12, pred_len=3):
        self.data = data          # (T, num_nodes)
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Total number of valid windows we can create
        # We need seq_len inputs + pred_len targets, so we stop early
        self.num_samples = len(data) - seq_len - pred_len + 1

        if self.num_samples <= 0:
            raise ValueError(
                f"Not enough data: {len(data)} timesteps for "
                f"seq_len={seq_len} + pred_len={pred_len}"
            )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns one (input, target) pair.

        x_seq : (seq_len, num_nodes, 1) — past traffic values
        y     : (num_nodes, pred_len)   — future traffic values to predict
        """
        # Input: timesteps [idx, idx+1, ..., idx+seq_len-1]
        x = self.data[idx : idx + self.seq_len]          # (seq_len, num_nodes)

        # Target: timesteps [idx+seq_len, ..., idx+seq_len+pred_len-1]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]  # (pred_len, num_nodes)

        # Reshape x to (seq_len, num_nodes, 1) — model expects 1 feature per node
        x_seq = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)  # add feature dim

        # Reshape y to (num_nodes, pred_len) — transpose so nodes are first
        y = torch.tensor(y.T, dtype=torch.float32)  # (num_nodes, pred_len)

        return x_seq, y


def create_datasets(traffic_data, seq_len=12, pred_len=3, scaler_path="ml/scaler.pkl"):
    """
    Split traffic data into train/val/test sets and normalize.

    The split is done BY TIME (not randomly!) because traffic is a time series.
    Shuffling would leak future information into training.

    Split: 70% train / 15% validation / 15% test

    Normalization: z-score using training data statistics only.
    This prevents data leakage — val/test stats are unknown during training.

    Args:
        traffic_data : numpy array of shape (T, num_nodes)
        seq_len      : input sequence length
        pred_len     : prediction horizon length
        scaler_path  : where to save normalization parameters

    Returns:
        train_dataset, val_dataset, test_dataset, scaler_dict
    """
    T = len(traffic_data)

    # --- Time-based split ---
    train_end = int(T * 0.70)   # first 70% for training
    val_end   = int(T * 0.85)   # next 15% for validation

    train_data = traffic_data[:train_end]
    val_data   = traffic_data[train_end:val_end]
    test_data  = traffic_data[val_end:]

    print(f"Data split (by time):")
    print(f"  Train: {len(train_data)} steps  (timesteps 0–{train_end - 1})")
    print(f"  Val:   {len(val_data)} steps  (timesteps {train_end}–{val_end - 1})")
    print(f"  Test:  {len(test_data)} steps  (timesteps {val_end}–{T - 1})")

    # --- Compute normalization stats from training data ONLY ---
    train_mean = train_data.mean()
    train_std  = train_data.std()

    # Prevent division by zero if data is constant (shouldn't happen)
    if train_std < 1e-8:
        train_std = 1.0

    print(f"Scaler: mean={train_mean:.4f}, std={train_std:.4f}")

    # --- Apply z-score normalization: (x - mean) / std ---
    train_norm = (train_data - train_mean) / train_std
    val_norm   = (val_data   - train_mean) / train_std
    test_norm  = (test_data  - train_mean) / train_std

    # --- Save scaler so we can un-normalize predictions later ---
    scaler = {"mean": float(train_mean), "std": float(train_std)}
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")

    # --- Create PyTorch datasets ---
    train_dataset = TrafficDataset(train_norm, seq_len, pred_len)
    val_dataset   = TrafficDataset(val_norm,   seq_len, pred_len)
    test_dataset  = TrafficDataset(test_norm,  seq_len, pred_len)

    print(f"Dataset sizes: train={len(train_dataset)}, "
          f"val={len(val_dataset)}, test={len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset, scaler


def load_scaler(scaler_path="ml/scaler.pkl"):
    """Load saved normalization parameters."""
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return scaler


def inverse_transform(data, scaler):
    """
    Undo z-score normalization: original = data * std + mean

    Args:
        data   : normalized numpy array or torch tensor
        scaler : dict with 'mean' and 'std' keys

    Returns:
        un-normalized data in the same format
    """
    return data * scaler["std"] + scaler["mean"]
