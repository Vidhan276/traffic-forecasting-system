"""
test_model.py — Unit tests for model architectures.

Tests:
  - Forward pass shapes for both tgcn_gcn and tgcn_gat
  - build_model factory
  - Model parameter counts are reasonable
  - Error on unknown model type
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.factory import build_model, list_models
from models.tgcn_gcn import TrafficGNN_GCN
from models.tgcn_gat import TrafficGNN_GAT


# ── Fixtures ──────────────────────────────────────────────────────────────────

NUM_NODES   = 50
SEQ_LEN     = 12
PRED_LEN    = 3
HIDDEN_DIM  = 16   # small for fast tests
NUM_HEADS   = 2    # small for fast tests


def _make_inputs():
    """Create random (seq_len, N, 1) input and random edge_index."""
    x_seq = torch.randn(SEQ_LEN, NUM_NODES, 1)
    # Random fully-connected subgraph (just a few edges for speed)
    src   = torch.randint(0, NUM_NODES, (NUM_NODES * 2,))
    dst   = torch.randint(0, NUM_NODES, (NUM_NODES * 2,))
    edge_index = torch.stack([src, dst], dim=0)
    return x_seq, edge_index


# ═══════════════════════════════════════════════════════════════════════════════
# TrafficGNN_GCN
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrafficGNN_GCN:
    def test_output_shape(self):
        model    = TrafficGNN_GCN(seq_len=SEQ_LEN, hidden_dim=HIDDEN_DIM, pred_len=PRED_LEN)
        x, ei    = _make_inputs()
        out      = model(x, ei)
        assert out.shape == (NUM_NODES, PRED_LEN), f"Unexpected shape: {out.shape}"

    def test_no_grad_runs(self):
        model = TrafficGNN_GCN(seq_len=SEQ_LEN, hidden_dim=HIDDEN_DIM, pred_len=PRED_LEN)
        x, ei = _make_inputs()
        with torch.no_grad():
            out = model(x, ei)
        assert out.requires_grad is False

    def test_parameters_exist(self):
        model  = TrafficGNN_GCN(hidden_dim=HIDDEN_DIM)
        params = sum(p.numel() for p in model.parameters())
        assert params > 0, "Model should have trainable parameters"

    def test_different_num_nodes(self):
        """GCNConv/GATConv are node-count-agnostic."""
        model = TrafficGNN_GCN(seq_len=SEQ_LEN, hidden_dim=HIDDEN_DIM, pred_len=PRED_LEN)
        for N in [10, 100, 500]:
            x  = torch.randn(SEQ_LEN, N, 1)
            ei = torch.randint(0, N, (2, N * 3))
            out = model(x, ei)
            assert out.shape == (N, PRED_LEN)


# ═══════════════════════════════════════════════════════════════════════════════
# TrafficGNN_GAT
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrafficGNN_GAT:
    def test_output_shape(self):
        model = TrafficGNN_GAT(
            seq_len=SEQ_LEN, hidden_dim=HIDDEN_DIM,
            pred_len=PRED_LEN, num_heads=NUM_HEADS,
        )
        x, ei = _make_inputs()
        out   = model(x, ei)
        assert out.shape == (NUM_NODES, PRED_LEN)

    def test_no_grad_runs(self):
        model = TrafficGNN_GAT(hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS)
        x, ei = _make_inputs()
        with torch.no_grad():
            out = model(x, ei)
        assert not out.requires_grad

    def test_parameters_more_than_gcn(self):
        """GAT should have more parameters than GCN (attention weights)."""
        gcn = TrafficGNN_GCN(hidden_dim=HIDDEN_DIM)
        gat = TrafficGNN_GAT(hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS)
        assert sum(p.numel() for p in gat.parameters()) > \
               sum(p.numel() for p in gcn.parameters())

    def test_different_num_nodes(self):
        model = TrafficGNN_GAT(hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS)
        for N in [10, 100]:
            x  = torch.randn(SEQ_LEN, N, 1)
            ei = torch.randint(0, N, (2, N * 3))
            out = model(x, ei)
            assert out.shape == (N, PRED_LEN)

    def test_pred_len_configurable(self):
        for pred_len in [1, 3, 6, 12]:
            model = TrafficGNN_GAT(hidden_dim=HIDDEN_DIM, pred_len=pred_len, num_heads=NUM_HEADS)
            x, ei = _make_inputs()
            out   = model(x, ei)
            assert out.shape[1] == pred_len


# ═══════════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════════

class TestFactory:
    def test_build_tgcn_gcn(self):
        cfg   = {"type": "tgcn_gcn", "hidden_dim": HIDDEN_DIM, "seq_len": SEQ_LEN,
                 "pred_len": PRED_LEN, "num_heads": 1}
        model = build_model(cfg)
        assert isinstance(model, TrafficGNN_GCN)

    def test_build_tgcn_gat(self):
        cfg   = {"type": "tgcn_gat", "hidden_dim": HIDDEN_DIM, "seq_len": SEQ_LEN,
                 "pred_len": PRED_LEN, "num_heads": NUM_HEADS}
        model = build_model(cfg)
        assert isinstance(model, TrafficGNN_GAT)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown model type"):
            build_model({"type": "unknown_model"})

    def test_list_models(self):
        models = list_models()
        assert "tgcn_gcn" in models
        assert "tgcn_gat" in models

    def test_factory_forward_pass(self):
        for model_type in list_models():
            cfg   = {"type": model_type, "hidden_dim": HIDDEN_DIM, "seq_len": SEQ_LEN,
                     "pred_len": PRED_LEN, "num_heads": NUM_HEADS}
            model = build_model(cfg)
            x, ei = _make_inputs()
            with torch.no_grad():
                out = model(x, ei)
            assert out.shape == (NUM_NODES, PRED_LEN), \
                f"Factory {model_type} produced wrong shape: {out.shape}"
