"""
loader.py — Central data-loading utilities.

All file-path resolution, config loading, and graph/data loading happens
here so that every other module has a single clean interface to call.

Usage:
    cfg          = load_config()                    # returns the config.yaml dict
    G, node_list = load_graph(cfg)                  # returns (networkx graph, node list)
    edge_index   = graph_to_edge_index(G)           # returns torch tensor (2, E)
    traffic_data = load_traffic(cfg)                # returns numpy array (T, N)
    scaler       = load_scaler(cfg)                 # returns {"mean": ..., "std": ...}
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


# ── Config ────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).parent.parent


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """
    Load and return the YAML configuration.

    Args:
        path: explicit path to config.yaml. Defaults to <project_root>/config.yaml.

    Returns:
        Nested dict with all configuration values.
    """
    if path is None:
        path = _PROJECT_ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(cfg: dict, key: str) -> Path:
    """Resolve a paths.* key from config relative to the project root."""
    return _PROJECT_ROOT / cfg["paths"][key]


# ── Graph Loading ──────────────────────────────────────────────────────────────

def load_graph(cfg: dict, key: str = "subgraph"):
    """
    Load a pickled NetworkX graph.

    Args:
        cfg: loaded config dict.
        key: which graph to load: "subgraph", "kothrud_graph", or "full_pune_graph".

    Returns:
        Tuple (G, node_list) where node_list preserves insertion order.
    """
    path = resolve_path(cfg, key)
    with open(path, "rb") as f:
        G = pickle.load(f)
        
    # ── Memory Optimization (Critical for Render 512MB limit) ────────────────
    # Strip unnecessary node attributes
    allowed_node_attrs = {"x", "y"}
    for node in G.nodes:
        node_data = G.nodes[node]
        keys_to_delete = [k for k in node_data if k not in allowed_node_attrs]
        for k in keys_to_delete:
            del node_data[k]
            
    # Strip unnecessary edge attributes (saves massive dictionary overhead for 320k edges)
    allowed_edge_attrs = {"length", "highway", "name", "geometry"}
    for u, v, k in G.edges(keys=True):
        edge_data = G.edges[u, v, k]
        keys_to_delete = [key_attr for key_attr in edge_data if key_attr not in allowed_edge_attrs]
        for key_attr in keys_to_delete:
            del edge_data[key_attr]
            
    import gc
    gc.collect()
    # ─────────────────────────────────────────────────────────────────────────

    node_list = list(G.nodes)
    return G, node_list


def graph_to_edge_index(G) -> torch.Tensor:
    """
    Convert a NetworkX graph to a PyTorch Geometric edge_index tensor.

    Strips all node and edge attributes first so PyG conversion is clean.

    Returns:
        edge_index tensor of shape (2, num_edges).
    """
    from torch_geometric.utils import from_networkx

    G_clean = G.copy()
    for node in G_clean.nodes:
        G_clean.nodes[node].clear()
    for u, v, k in G_clean.edges(keys=True):
        G_clean.edges[u, v, k].clear()

    data = from_networkx(G_clean)
    return data.edge_index


# ── Traffic Data Loading ───────────────────────────────────────────────────────

def load_traffic(cfg: dict, key: str = "traffic_data") -> np.ndarray:
    """
    Load the full traffic numpy array.

    Args:
        cfg: loaded config dict.
        key: paths key for the data file.

    Returns:
        numpy array of shape (T, num_nodes), dtype float32.
    """
    path = resolve_path(cfg, key)
    return np.load(str(path))


def load_last_sequence(cfg: dict, seq_len: int | None = None) -> np.ndarray:
    """
    Load only the last ``seq_len`` timesteps using memory-mapped access.

    This avoids loading the full ~1 GB city-wide data file into RAM.

    Returns:
        numpy array of shape (seq_len, num_nodes).
    """
    if seq_len is None:
        seq_len = cfg["model"]["seq_len"]

    # Try city-wide data first, fall back to subgraph data
    for key in ("full_traffic_data", "traffic_data"):
        path = resolve_path(cfg, key)
        if path.exists():
            data = np.load(str(path), mmap_mode="r")
            return np.array(data[-seq_len:])

    raise FileNotFoundError(
        "Neither traffic_data nor full_traffic_data found. "
        "Run data/generate_data.py first."
    )


# ── Scaler Loading ────────────────────────────────────────────────────────────

def load_scaler(cfg: dict) -> dict[str, float]:
    """
    Load the z-score scaler saved during training.

    Returns:
        Dict with 'mean' and 'std' keys (both Python floats).
    """
    path = resolve_path(cfg, "scaler")
    with open(path, "rb") as f:
        return pickle.load(f)
