"""
forecasting.py — ForecastingService

Loads the trained T-GCN model ONCE at startup and exposes a fast
``predict()`` method that runs inference without reloading anything.

Design decisions:
  - Model is loaded in ``load()``, stored on ``self.model`` — never reloaded.
  - ``torch.no_grad()`` wraps the entire inference path.
  - ``mmap_mode='r'`` reads only the last 12 timesteps from the large data file,
    keeping RAM usage near zero for the 1 GB city-wide dataset.
  - All tensors stay on CPU (no unnecessary GPU transfers for a small model).

Usage:
    svc = ForecastingService(config)
    svc.load()                          # call once at server startup
    preds = svc.predict()               # (num_nodes, pred_len), unnormalised
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ForecastingService:
    """
    Wraps the trained GNN model for production inference.

    Attributes:
        cfg         : full config dict from config.yaml.
        model       : loaded PyTorch model (set after calling load()).
        edge_index  : graph connectivity tensor (set after calling load()).
        scaler      : normalisation parameters dict.
        node_list   : ordered list of OSM node IDs.
    """

    def __init__(self, cfg: dict[str, Any]):
        self.cfg        = cfg
        self.model      = None
        self.edge_index = None
        self.scaler     = None
        self.node_list  = None
        self._device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Load model, graph, and scaler.  Call this ONCE at application startup.

        Raises:
            FileNotFoundError: if model weights or graph file are missing.
        """
        import sys
        from pathlib import Path as P

        # Ensure project root is on the Python path so ml/ imports work
        root = P(__file__).parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        if str(root / "ml") not in sys.path:
            sys.path.insert(0, str(root / "ml"))

        from data.loader import load_graph, graph_to_edge_index, load_scaler
        from models.factory import build_model

        logger.info("ForecastingService: loading graph...")
        G, self.node_list = load_graph(self.cfg, key="subgraph")

        self.edge_index = graph_to_edge_index(G).to(self._device)
        logger.info(
            "Graph loaded: %d nodes, %d edges",
            len(self.node_list),
            self.edge_index.shape[1],
        )

        logger.info("ForecastingService: loading model...")
        self.model = build_model(self.cfg["model"]).to(self._device)

        weights_path = root / self.cfg["paths"]["model_weights"]
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {weights_path}. "
                "Run ml/train_model.py first."
            )
        self.model.load_state_dict(
            torch.load(str(weights_path), map_location=self._device, weights_only=True)
        )
        self.model.eval()

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info("Model loaded (%s): %d parameters", self.cfg["model"]["type"], n_params)

        logger.info("ForecastingService: loading scaler...")
        self.scaler = load_scaler(self.cfg)
        logger.info("Scaler: mean=%.4f, std=%.4f", self.scaler["mean"], self.scaler["std"])

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        traffic_seq: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Run the model and return per-node traffic forecasts.

        Args:
            traffic_seq: (seq_len, num_nodes) array of recent traffic values
                         in the normalised space [approximately -3 to +3].
                         If None, reads the last seq_len timesteps from disk.

        Returns:
            numpy array of shape (num_nodes, pred_len) with **unnormalised**
            traffic values in the original [0, 1] range.
        """
        if self.model is None:
            raise RuntimeError("Call ForecastingService.load() before predict().")

        if traffic_seq is None:
            traffic_seq = self._load_last_sequence()

        # Normalise if not already done (values > 2 suggest raw [0,1] data)
        if float(np.abs(traffic_seq).max()) > 2.0:
            traffic_seq = (traffic_seq - self.scaler["mean"]) / self.scaler["std"]

        # (seq_len, num_nodes) → (seq_len, num_nodes, 1)
        x = torch.tensor(traffic_seq, dtype=torch.float32).unsqueeze(-1).to(self._device)

        with torch.no_grad():
            pred = self.model(x, self.edge_index)  # (num_nodes, pred_len)

        # Un-normalise back to original traffic scale
        pred_np = pred.cpu().numpy()
        pred_unnorm = pred_np * self.scaler["std"] + self.scaler["mean"]

        return pred_unnorm  # (num_nodes, pred_len)

    def get_current_traffic(self) -> np.ndarray:
        """
        Return the most recent traffic observation for each node.

        Returns:
            numpy array of shape (num_nodes,) with un-normalised values.
        """
        seq = self._load_last_sequence()
        # Last timestep, un-normalised
        return seq[-1] * self.scaler["std"] + self.scaler["mean"]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_last_sequence(self) -> np.ndarray:
        """Read last seq_len timesteps efficiently via memory-map."""
        from data.loader import load_last_sequence
        key = "full_traffic_data" if len(self.node_list) > 5000 else "traffic_data"
        raw = load_last_sequence(self.cfg, key=key)
        # Normalise
        return (raw - self.scaler["mean"]) / self.scaler["std"]
