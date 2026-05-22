"""
factory.py — Model factory.

Single function that reads the model type from config and returns the
correctly instantiated model.  Add new architectures here.

Usage:
    config = load_config()
    model  = build_model(config["model"])
"""

import torch.nn as nn

from .tgcn_gcn import TrafficGNN_GCN
from .tgcn_gat import TrafficGNN_GAT


# Registry mapping name → class.  Add new models here.
_MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "tgcn_gcn": TrafficGNN_GCN,
    "tgcn_gat": TrafficGNN_GAT,
}


def build_model(model_cfg: dict) -> nn.Module:
    """
    Instantiate a traffic forecasting model from a config dict.

    The config dict is the ``model:`` section of config.yaml:
        {
          "type":       "tgcn_gat",
          "hidden_dim": 32,
          "num_heads":  4,
          "seq_len":    12,
          "pred_len":   3,
        }

    Args:
        model_cfg: dict with keys type, hidden_dim, num_heads, seq_len, pred_len.

    Returns:
        Instantiated (untrained) nn.Module.

    Raises:
        ValueError: if ``type`` is not in the model registry.
    """
    model_type = model_cfg.get("type", "tgcn_gat")

    if model_type not in _MODEL_REGISTRY:
        available = list(_MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Available options: {available}"
        )

    model_cls = _MODEL_REGISTRY[model_type]

    return model_cls(
        seq_len=model_cfg.get("seq_len", 12),
        hidden_dim=model_cfg.get("hidden_dim", 32),
        pred_len=model_cfg.get("pred_len", 3),
        num_heads=model_cfg.get("num_heads", 4),
    )


def list_models() -> list[str]:
    """Return the names of all registered model types."""
    return list(_MODEL_REGISTRY.keys())
