"""
tgcn_gat.py — GAT-enhanced T-GCN (primary upgraded model).

Architecture:
  GATConv × 2  →  GRUCell × seq_len  →  Linear

Key improvement over the GCN version:
  GATConv learns attention weights for each neighbour, so the model can
  automatically assign higher importance to major roads (high connectivity)
  and lower importance to residential feeders.  The rest of the pipeline —
  GRUCell for temporal modelling and a linear prediction head — is identical.

The forward() signature is unchanged from TrafficGNN_GCN, so this is a
drop-in replacement inside train_model.py and the FastAPI service.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import PurePyTorchGATConv as GATConv


class TrafficGNN_GAT(nn.Module):
    """
    GAT-enhanced T-GCN: Temporal Graph Attention Network.

    Each GATConv layer computes a weighted sum of neighbour features where
    the weights are learned via a small attention MLP.

    Multi-head attention (``num_heads`` heads) is used in the first GAT layer
    and the outputs are concatenated, doubling the feature size.  A linear
    projection immediately reduces it back to ``hidden_dim`` so the GRUCell
    input size stays fixed.

    Args:
        seq_len    (int): input timesteps (default 12 = 1 hour @ 5-min intervals).
        hidden_dim (int): hidden feature size for both GAT and GRU layers.
        pred_len   (int): future timesteps to predict (default 3 = 15 min).
        num_heads  (int): number of GAT attention heads (default 4).
    """

    name = "tgcn_gat"

    def __init__(
        self,
        seq_len: int = 12,
        hidden_dim: int = 32,
        pred_len: int = 3,
        num_heads: int = 4,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.pred_len = pred_len
        self.num_heads = num_heads

        # ── Spatial layers: two GAT layers ──────────────────────────────────
        # First GAT: 1 input feature → hidden_dim (with multi-head concat)
        # concat=True: output size = hidden_dim * num_heads
        self.gat1 = GATConv(
            in_channels=1,
            out_channels=hidden_dim,
            heads=num_heads,
            concat=True,    # concatenate heads → output dim = hidden_dim * num_heads
            dropout=0.0,    # keep 0 for inference stability on small graphs
        )
        # Project concatenated heads back to hidden_dim
        self.proj1 = nn.Linear(hidden_dim * num_heads, hidden_dim)

        # Second GAT: hidden_dim → hidden_dim (single head, average aggregation)
        self.gat2 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=1,
            concat=False,   # average over 1 head → output dim = hidden_dim
            dropout=0.0,
        )

        # ── Temporal layer: GRU ──────────────────────────────────────────────
        # Processes spatial features one timestep at a time
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # ── Output layer ─────────────────────────────────────────────────────
        self.fc = nn.Linear(hidden_dim, pred_len)

    def forward(
        self,
        x_seq: torch.Tensor,      # (seq_len, num_nodes, 1)
        edge_index: torch.Tensor,  # (2, num_edges)
    ) -> torch.Tensor:             # (num_nodes, pred_len)
        """
        Process a sequence of graph snapshots and forecast future traffic.

        Args:
            x_seq      : past traffic sequence, shape (seq_len, num_nodes, 1).
            edge_index : COO graph connectivity, shape (2, num_edges).

        Returns:
            Predicted traffic for next ``pred_len`` timesteps,
            shape (num_nodes, pred_len).
        """
        num_nodes = x_seq.size(1)

        # Initialise GRU hidden state to zeros
        h = torch.zeros(num_nodes, self.hidden_dim, device=x_seq.device)

        for t in range(self.seq_len):
            x = x_seq[t]   # (num_nodes, 1)

            # ── Spatial: GAT layer 1 ──────────────────────────────────────
            x = F.elu(self.gat1(x, edge_index))    # (num_nodes, hidden_dim * heads)
            x = F.elu(self.proj1(x))               # (num_nodes, hidden_dim)

            # ── Spatial: GAT layer 2 ──────────────────────────────────────
            x = F.elu(self.gat2(x, edge_index))    # (num_nodes, hidden_dim)

            # ── Temporal: GRU update ──────────────────────────────────────
            h = self.gru(x, h)                     # (num_nodes, hidden_dim)

        return self.fc(h)   # (num_nodes, pred_len)
