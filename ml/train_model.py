"""
train_model.py — Config-driven, model-agnostic training pipeline.

Reads all settings from config.yaml so you never need to edit this file
to change hyperparameters — just update config.yaml.

Pipeline:
  1. Load config, graph, and traffic data.
  2. Build the model chosen in config.model.type.
  3. Train with early stopping; save best checkpoint.
  4. Print final metrics.

Run from project root:
    py -3.13 ml/train_model.py
    py -3.13 ml/train_model.py --quick   (50 epochs, for fast testing)
    py -3.13 ml/train_model.py --model tgcn_gcn  (override model type)
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── Path setup (run from project root) ───────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import TrafficDataset, create_datasets, compute_metrics
from data.loader import load_config, load_graph, graph_to_edge_index, load_traffic
from models.factory import build_model


def train(cfg: dict, quick: bool = False, model_override: str | None = None) -> None:
    """Run the full training pipeline."""

    # ── Override from CLI ─────────────────────────────────────────────────────
    if model_override:
        cfg["model"]["type"] = model_override

    model_cfg  = cfg["model"]
    train_cfg  = cfg["training"]
    paths_cfg  = cfg["paths"]

    SEQ_LEN    = model_cfg["seq_len"]
    PRED_LEN   = model_cfg["pred_len"]
    EPOCHS     = 50 if quick else train_cfg["epochs"]
    PATIENCE   = train_cfg["patience"]
    LR         = train_cfg["lr"]
    BATCH      = train_cfg["batch_sample"]
    VAL_BATCH  = train_cfg["val_sample"]
    MODEL_PATH = ROOT / paths_cfg["model_weights"]
    HISTORY_PATH = ROOT / paths_cfg["training_history"]
    MAPE_EPS   = cfg.get("metrics", {}).get("mape_epsilon", 0.1)

    print("=" * 60)
    print(f"  T-GCN Training  -  model: {model_cfg['type']}")
    print("=" * 60)

    # ── 1. Load graph ──────────────────────────────────────────────────────────
    print("\nLoading graph...")
    G, node_list = load_graph(cfg, key="subgraph")
    num_nodes = len(node_list)
    print(f"  Graph: {num_nodes} nodes, {len(G.edges)} edges")

    edge_index = graph_to_edge_index(G)

    # ── 2. Load traffic data ──────────────────────────────────────────────────
    print("Loading traffic data...")
    traffic_data = load_traffic(cfg)
    assert traffic_data.shape[1] == num_nodes, (
        f"Mismatch: traffic has {traffic_data.shape[1]} nodes, graph has {num_nodes}"
    )
    print(f"  Shape: {traffic_data.shape}")

    # ── 3. Create datasets ────────────────────────────────────────────────────
    print("\nCreating datasets...")
    scaler_path = str(ROOT / paths_cfg["scaler"])
    train_ds, val_ds, test_ds, scaler = create_datasets(
        traffic_data, seq_len=SEQ_LEN, pred_len=PRED_LEN, scaler_path=scaler_path
    )

    # ── 4. Build model ────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model      = build_model(model_cfg).to(device)
    edge_index = edge_index.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_cfg['type']}  |  parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn   = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=train_cfg["lr_factor"],
        patience=train_cfg["lr_patience"],
    )

    # ── 5. Training loop ──────────────────────────────────────────────────────
    print(f"\nTraining: epochs={EPOCHS}, patience={PATIENCE}")
    print("-" * 60)

    best_val_loss    = float("inf")
    patience_counter = 0
    history          = {
        "train_loss": [], "val_loss": [],
        "val_mae": [], "val_rmse": [], "val_mape": [],
    }

    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):

        # ── Train ──
        model.train()
        epoch_loss = 0.0
        indices    = random.sample(range(len(train_ds)), min(BATCH, len(train_ds)))

        for i in indices:
            x_seq, y = train_ds[i]
            x_seq, y = x_seq.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(x_seq, edge_index)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train = epoch_loss / len(indices)

        # ── Validate ──
        model.eval()
        val_preds, val_targets = [], []
        val_loss = 0.0
        val_idx  = random.sample(range(len(val_ds)), min(VAL_BATCH, len(val_ds)))

        with torch.no_grad():
            for i in val_idx:
                x_seq, y = val_ds[i]
                x_seq = x_seq.to(device)
                pred  = model(x_seq, edge_index)
                val_loss += loss_fn(pred, y.to(device)).item()
                val_preds.append(pred.cpu().numpy())
                val_targets.append(y.numpy())

        avg_val  = val_loss / len(val_idx)
        val_m    = compute_metrics(
            np.concatenate(val_preds),
            np.concatenate(val_targets),
            mape_eps=MAPE_EPS,
        )

        scheduler.step(avg_val)

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_mae"].append(val_m["mae"])
        history["val_rmse"].append(val_m["rmse"])
        history["val_mape"].append(val_m["robust_mape"])

        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:3d}/{EPOCHS} | "
                f"Train: {avg_train:.5f} | "
                f"Val: {avg_val:.5f} | "
                f"MAE: {val_m['mae']:.4f} | "
                f"R2: {val_m['r2']:.4f} | "
                f"LR: {lr_now:.5f}"
            )

        # ── Early stopping ──
        if avg_val < best_val_loss:
            best_val_loss    = avg_val
            patience_counter = 0
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(MODEL_PATH))
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (patience={PATIENCE})")
                break

    elapsed = time.time() - t_start

    # ── 6. Save history ───────────────────────────────────────────────────────
    import pickle
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(str(HISTORY_PATH), "wb") as f:
        pickle.dump(history, f)

    # ── 7. Final evaluation on test set ──────────────────────────────────────
    model.load_state_dict(torch.load(str(MODEL_PATH), map_location=device, weights_only=True))
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for i in range(len(test_ds)):
            x_seq, y = test_ds[i]
            pred = model(x_seq.to(device), edge_index)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.numpy())

    test_m = compute_metrics(
        np.concatenate(all_preds),
        np.concatenate(all_targets),
        mape_eps=MAPE_EPS,
    )

    # Save test metrics as pickle (for existing app.py) and JSON (for comparison)
    import pickle, json
    metrics_pkl_path = ROOT / paths_cfg["test_metrics"]
    with open(str(metrics_pkl_path), "wb") as f:
        pickle.dump(test_m, f)

    metrics_json_path = metrics_pkl_path.with_suffix(".json")
    with open(str(metrics_json_path), "w") as f:
        json.dump(test_m, f, indent=2)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Time:          {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Model:         {model_cfg['type']}")
    print(f"  Parameters:    {n_params:,}")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Test MAE:      {test_m['mae']:.4f}")
    print(f"  Test RMSE:     {test_m['rmse']:.4f}")
    print(f"  Test Robust MAPE: {test_m['robust_mape']:.2f}%")
    print(f"  Test R2:       {test_m['r2']:.4f}")
    print(f"  Saved ->        {MODEL_PATH}")
    print("=" * 60)
    print(f"\nNext step:  py -3.13 -m uvicorn api.main:app --port 8000")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the traffic forecasting model.")
    parser.add_argument("--quick", action="store_true", help="Run only 50 epochs (for testing).")
    parser.add_argument("--model", type=str, default=None, help="Override model type.")
    args = parser.parse_args()

    cfg = load_config()
    train(cfg, quick=args.quick, model_override=args.model)