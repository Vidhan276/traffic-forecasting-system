"""
evaluate.py — Model evaluation with ablation study.

Evaluates the trained model on the test set and compares it against:
  1. Naive last-value baseline (repeat last observation for all future steps).
  2. LSTM-only baseline (no graph — just per-node LSTM on traffic series).

Outputs:
  - Formatted terminal table
  - ml/test_metrics.json     (model metrics)
  - ml/ablation_results.json (comparison of all methods)
  - visualization/forecast_comparison.png

Run from project root:
    py -3.13 ml/evaluate.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import create_datasets, compute_metrics, save_metrics_json, inverse_transform
from data.loader import load_config, load_graph, graph_to_edge_index, load_traffic
from models.factory import build_model


# ══════════════════════════════════════════════════════════════════════════════
# Baseline models
# ══════════════════════════════════════════════════════════════════════════════

def naive_last_value_predict(dataset) -> tuple[np.ndarray, np.ndarray]:
    """
    Naive baseline: predict the last observed value for every future step.
    Shape: predictions = targets shape = (N_samples × N_nodes × pred_len).
    """
    preds, targets = [], []
    for i in range(len(dataset)):
        x_seq, y = dataset[i]
        # Last timestep: (num_nodes, 1) → squeeze to (num_nodes,)
        last_val = x_seq[-1, :, 0].numpy()           # (num_nodes,)
        pred_len = y.shape[1]
        # Tile last value across pred_len
        pred = np.tile(last_val[:, np.newaxis], (1, pred_len))  # (num_nodes, pred_len)
        preds.append(pred)
        targets.append(y.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(targets, axis=0)


class LSTMBaseline(nn.Module):
    """
    LSTM-only baseline — no graph, just an LSTM on per-node time series.

    Processes each node independently (no spatial information at all).
    Useful for ablation: quantifies how much the GCN/GAT layers contribute.
    """
    def __init__(self, seq_len: int = 12, hidden_dim: int = 32, pred_len: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim,
                            num_layers=2, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, pred_len)

    def forward(self, x_seq: torch.Tensor, edge_index=None) -> torch.Tensor:
        """x_seq: (seq_len, num_nodes, 1)."""
        # Rearrange to (num_nodes, seq_len, 1) for batch_first LSTM
        x = x_seq.permute(1, 0, 2)           # (N, seq_len, 1)
        out, _ = self.lstm(x)                 # (N, seq_len, hidden)
        return self.fc(out[:, -1, :])         # (N, pred_len)


def train_lstm_baseline(train_ds, val_ds, edge_index, device, epochs=50) -> LSTMBaseline:
    """Quick training of the LSTM baseline for ablation."""
    import random
    model     = LSTMBaseline().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn   = nn.MSELoss()
    best_val  = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        for i in random.sample(range(len(train_ds)), min(64, len(train_ds))):
            x, y = train_ds[i]
            optimizer.zero_grad()
            loss_fn(model(x.to(device)), y.to(device)).backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in random.sample(range(len(val_ds)), min(32, len(val_ds))):
                x, y = val_ds[i]
                val_loss += loss_fn(model(x.to(device)), y.to(device)).item()
        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model


def eval_model_on_test(model, test_ds, edge_index, device) -> tuple[np.ndarray, np.ndarray]:
    """Run model over every test sample and collect predictions + targets."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for i in range(len(test_ds)):
            x_seq, y = test_ds[i]
            pred = model(x_seq.to(device), edge_index)
            preds.append(pred.cpu().numpy())
            targets.append(y.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(targets, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    cfg       = load_config()
    model_cfg = cfg["model"]
    paths_cfg = cfg["paths"]
    MAPE_EPS  = cfg.get("metrics", {}).get("mape_epsilon", 0.1)

    print("=" * 60)
    print(f"  Evaluation - model: {model_cfg['type']}")
    print("=" * 60)

    # ── Load ──
    G, node_list = load_graph(cfg, key="subgraph")
    edge_index   = graph_to_edge_index(G)
    traffic_data = load_traffic(cfg)
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler_path = str(ROOT / paths_cfg["scaler"])
    train_ds, val_ds, test_ds, scaler = create_datasets(
        traffic_data,
        seq_len=model_cfg["seq_len"],
        pred_len=model_cfg["pred_len"],
        scaler_path=scaler_path,
    )

    edge_index = edge_index.to(device)

    # ── Main model ────────────────────────────────────────────────────────────
    model = build_model(model_cfg).to(device)
    model_path = ROOT / paths_cfg["model_weights"]
    model.load_state_dict(
        torch.load(str(model_path), map_location=device, weights_only=True)
    )

    print(f"\nRunning {model_cfg['type']} on test set...")
    preds, targets = eval_model_on_test(model, test_ds, edge_index, device)
    main_metrics   = compute_metrics(preds, targets, mape_eps=MAPE_EPS)

    # ── Baselines ─────────────────────────────────────────────────────────────
    print("Running naive last-value baseline...")
    lv_preds, lv_targets = naive_last_value_predict(test_ds)
    lv_metrics = compute_metrics(lv_preds, lv_targets, mape_eps=MAPE_EPS)

    print("Training LSTM baseline (50 epochs)...")
    lstm_model  = train_lstm_baseline(train_ds, val_ds, edge_index, device, epochs=50)
    lstm_preds, lstm_targets = eval_model_on_test(lstm_model, test_ds, None, device)
    lstm_metrics = compute_metrics(lstm_preds, lstm_targets, mape_eps=MAPE_EPS)

    # ── Print comparison table ─────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print(f"  {'Model':<22} {'MAE':>8} {'RMSE':>8} {'R-MAPE%':>10} {'R2':>8}")
    print("  " + "-" * 64)

    def _row(name, m):
        print(
            f"  {name:<22} "
            f"{m['mae']:>8.4f} "
            f"{m['rmse']:>8.4f} "
            f"{m['robust_mape']:>9.2f}% "
            f"{m['r2']:>8.4f}"
        )

    _row("Last-value baseline", lv_metrics)
    _row("LSTM baseline",       lstm_metrics)
    _row(model_cfg["type"],     main_metrics)
    print("=" * 68)

    # ── Save metrics ─────────────────────────────────────────────────────────
    test_pkl = ROOT / paths_cfg["test_metrics"]
    with open(str(test_pkl), "wb") as f:
        pickle.dump(main_metrics, f)

    save_metrics_json(main_metrics, str(test_pkl.with_suffix(".json")))

    ablation = {
        "last_value_baseline": lv_metrics,
        "lstm_baseline":       lstm_metrics,
        model_cfg["type"]:     main_metrics,
    }
    abl_path = ROOT / paths_cfg.get("ablation_results", "ml/ablation_results.json")
    save_metrics_json(ablation, str(abl_path))

    # ── Forecast comparison plot ──────────────────────────────────────────────
    plot_path = ROOT / "visualization" / "forecast_comparison.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    n_plot = min(100, len(test_ds))
    num_nodes = len(node_list)
    node_indices = np.linspace(0, num_nodes - 1, 4, dtype=int)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(
        f"Forecast vs Actual - {model_cfg['type'].upper()} (Test Set)",
        fontsize=14, fontweight=600,
    )

    for ax_idx, node_idx in enumerate(node_indices):
        ax = axes[ax_idx // 2, ax_idx % 2]
        
        # Extract predictions/targets for this specific node across n_plot samples
        # shape is (n_samples * num_nodes, pred_len)
        p_seq = np.array([preds[s * num_nodes + node_idx, 0] for s in range(n_plot)])
        t_seq = np.array([targets[s * num_nodes + node_idx, 0] for s in range(n_plot)])
        
        p = inverse_transform(p_seq, scaler)
        t = inverse_transform(t_seq, scaler)
        ts = np.arange(len(p)) * 5

        ax.plot(ts, t, color="#1A73E8", lw=1.5, label="Actual")
        ax.plot(ts, p, color="#EA4335", lw=1.5, ls="--", alpha=0.85, label="Predicted")
        ax.set_title(f"Node {node_idx}", fontsize=11)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Traffic Level")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved -> {plot_path}")
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
