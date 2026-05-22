"""
tgcn_gcn.py — Original T-GCN using GCNConv (kept for ablation studies).

Architecture:
  GCNConv × 2  →  GRUCell × seq_len  →  Linear
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class TrafficGNN_GCN(nn.Module):
    """
    T-GCN: Temporal Graph Convolutional Network (GCN variant).

    This is the original architecture, kept for ablation comparison against
    the GAT-enhanced version. The two GCNConv layers treat all neighbours
    equally (uniform weight based on degree normalisation).

    Args:
        seq_len    (int): input timesteps, e.g. 12 → 1 hour of history.
        hidden_dim (int): feature dimension for GCN and GRU layers.
        pred_len   (int): future timesteps to predict, e.g. 3 → 15 min ahead.
        num_heads  (int): unused; included for API compatibility with GAT version.
    """

    name = "tgcn_gcn"

    def __init__(
        self,
        seq_len: int = 12,
        hidden_dim: int = 32,
        pred_len: int = 3,
        num_heads: int = 1,  # unused, for factory compatibility
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.pred_len = pred_len

        # Spatial: two GCN layers
        self.gc1 = GCNConv(1, hidden_dim)
        self.gc2 = GCNConv(hidden_dim, hidden_dim)

        # Temporal: GRU processes one timestep at a time
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # Output: predict pred_len future values per node
        self.fc = nn.Linear(hidden_dim, pred_len)

    def forward(
        self,
        x_seq: torch.Tensor,     # (seq_len, num_nodes, 1)
        edge_index: torch.Tensor, # (2, num_edges)
    ) -> torch.Tensor:            # (num_nodes, pred_len)
        """Process a sequence of graph snapshots and forecast future traffic."""
        num_nodes = x_seq.size(1)
        h = torch.zeros(num_nodes, self.hidden_dim, device=x_seq.device)

        for t in range(self.seq_len):
            x = x_seq[t]                              # (num_nodes, 1)
            x = F.relu(self.gc1(x, edge_index))       # (num_nodes, hidden_dim)
            x = F.relu(self.gc2(x, edge_index))       # (num_nodes, hidden_dim)
            h = self.gru(x, h)                        # (num_nodes, hidden_dim)

        return self.fc(h)                             # (num_nodes, pred_len)
