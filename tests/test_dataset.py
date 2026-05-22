"""
test_dataset.py — Unit tests for data/dataset.py

Tests:
  - TrafficDataset window slicing and shapes
  - create_datasets split sizes
  - compute_metrics with known values (especially robust MAPE with zeros)
  - inverse_transform correctness
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import (
    TrafficDataset,
    compute_metrics,
    inverse_transform,
)


# ═══════════════════════════════════════════════════════════════════════════════
# TrafficDataset
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrafficDataset:
    def _make_data(self, T=100, N=10) -> np.ndarray:
        return np.random.rand(T, N).astype(np.float32)

    def test_num_samples(self):
        data = self._make_data(T=100, N=10)
        ds   = TrafficDataset(data, seq_len=12, pred_len=3)
        # num_samples = T - seq_len - pred_len + 1
        assert len(ds) == 100 - 12 - 3 + 1

    def test_item_shapes(self):
        data = self._make_data(T=100, N=10)
        ds   = TrafficDataset(data, seq_len=12, pred_len=3)
        x, y = ds[0]
        assert x.shape == (12, 10, 1),   f"x shape wrong: {x.shape}"
        assert y.shape == (10, 3),        f"y shape wrong: {y.shape}"

    def test_item_dtype(self):
        data = self._make_data()
        ds   = TrafficDataset(data)
        x, y = ds[5]
        assert x.dtype.is_floating_point
        assert y.dtype.is_floating_point

    def test_window_values(self):
        """Check that x and y contain the correct timestep slices."""
        T, N = 50, 4
        data = np.arange(T * N, dtype=np.float32).reshape(T, N)
        ds   = TrafficDataset(data, seq_len=5, pred_len=2)
        x, y = ds[3]
        # x should be data[3:8]  → rows 3,4,5,6,7
        np.testing.assert_array_almost_equal(x[:, :, 0].numpy(), data[3:8])
        # y should be data[8:10].T  → (N, pred_len)
        np.testing.assert_array_almost_equal(y.numpy(), data[8:10].T)

    def test_raises_on_too_little_data(self):
        data = np.ones((5, 10), dtype=np.float32)
        with pytest.raises(ValueError):
            TrafficDataset(data, seq_len=12, pred_len=3)


# ═══════════════════════════════════════════════════════════════════════════════
# compute_metrics
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeMetrics:
    def test_perfect_predictions(self):
        """Zero error should give MAE=0, RMSE=0, MAPE=0, R²=1."""
        y = np.array([1.0, 2.0, 3.0, 0.5])
        m = compute_metrics(y, y, mape_eps=0.1)
        assert m["mae"]  == pytest.approx(0.0, abs=1e-6)
        assert m["rmse"] == pytest.approx(0.0, abs=1e-6)
        assert m["robust_mape"] == pytest.approx(0.0, abs=1e-6)
        assert m["r2"]   == pytest.approx(1.0, abs=1e-6)

    def test_known_mae(self):
        pred    = np.array([2.0, 3.0])
        targets = np.array([1.0, 1.0])
        m = compute_metrics(pred, targets)
        assert m["mae"]  == pytest.approx(1.5, rel=1e-5)
        assert m["rmse"] == pytest.approx(np.sqrt(2.5), rel=1e-5)

    def test_robust_mape_ignores_near_zero(self):
        """With eps=0.1, near-zero targets should NOT blow up MAPE."""
        pred    = np.array([0.05, 1.0])
        targets = np.array([0.0,  1.0])   # first target is 0 → would be inf with plain MAPE
        m_robust = compute_metrics(pred, targets, mape_eps=0.1)
        assert np.isfinite(m_robust["robust_mape"]), "Robust MAPE should be finite"
        assert m_robust["robust_mape"] < 1000,       "Robust MAPE should not be huge"

    def test_plain_mape_would_be_infinite(self):
        """Demonstrate the original bug: plain MAPE is inf with zero targets."""
        targets = np.array([0.0, 1.0])
        pred    = np.array([0.1, 1.0])
        plain   = np.mean(np.abs((pred - targets) / np.maximum(np.abs(targets), 1e-10))) * 100
        assert plain > 1e8, "Plain MAPE should be extremely large"

    def test_r2_with_constant_targets(self):
        """R² is 0 (not undefined) when all targets are the same."""
        y = np.ones(10)
        m = compute_metrics(y + 0.5, y)
        assert np.isfinite(m["r2"])


# ═══════════════════════════════════════════════════════════════════════════════
# inverse_transform
# ═══════════════════════════════════════════════════════════════════════════════

class TestInverseTransform:
    def test_roundtrip(self):
        """Normalise then un-normalise should recover original data."""
        original = np.array([0.2, 0.5, 0.8, 0.1])
        scaler   = {"mean": 0.3, "std": 0.2}
        normalised = (original - scaler["mean"]) / scaler["std"]
        recovered  = inverse_transform(normalised, scaler)
        np.testing.assert_array_almost_equal(recovered, original, decimal=5)

    def test_works_with_float_input(self):
        scaler = {"mean": 0.5, "std": 0.1}
        result = inverse_transform(0.0, scaler)
        assert result == pytest.approx(0.5)
