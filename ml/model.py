"""
model.py — T-GCN: Temporal Graph Convolutional Network for traffic forecasting.

Architecture:
  1. GCN (Graph Convolutional Network) captures SPATIAL relationships
     between road intersections — nearby roads influence each other.
  2. GRU (Gated Recurrent Unit) captures TEMPORAL patterns —
     traffic changes over time (rush hours, daily cycles).

For each timestep in the input sequence:
  - GCN processes the traffic snapshot across all nodes
  - GRU updates its hidden state with the spatial features
After processing all timesteps, a linear layer predicts the future.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class TrafficGNN(nn.Module):
    """
    T-GCN: Temporal Graph Convolutional Network

    Args:
        seq_len  (int): number of past timesteps as input  (e.g. 12 = 1 hour)
        hidden_dim (int): size of hidden layer              (e.g. 32)
        pred_len (int): number of future timesteps to predict (e.g. 3 = 15 min)
    """

    def __init__(self, seq_len=12, hidden_dim=32, pred_len=3):
        super(TrafficGNN, self).__init__()

        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.pred_len = pred_len

        # --- Spatial layers: two GCN layers ---
        # First GCN: 1 input feature (traffic value) -> hidden_dim
        self.gc1 = GCNConv(1, hidden_dim)
        # Second GCN: refine spatial features
        self.gc2 = GCNConv(hidden_dim, hidden_dim)

        # --- Temporal layer: GRU cell ---
        # GRUCell processes one timestep at a time
        # Input size = hidden_dim (from GCN output)
        # Hidden size = hidden_dim (recurrent state)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # --- Output layer ---
        # Maps hidden state to pred_len future predictions
        self.fc = nn.Linear(hidden_dim, pred_len)

    def forward(self, x_seq, edge_index):
        """
        Forward pass: process a sequence of traffic snapshots.

        Args:
            x_seq      : (seq_len, num_nodes, 1) — past traffic values
            edge_index : (2, num_edges) — graph connectivity

        Returns:
            out : (num_nodes, pred_len) — predicted future traffic
        """
        num_nodes = x_seq.size(1)

        # Initialize GRU hidden state to zeros
        # Shape: (num_nodes, hidden_dim) — one hidden vector per node
        h = torch.zeros(num_nodes, self.hidden_dim, device=x_seq.device)

        # Process each timestep sequentially
        for t in range(self.seq_len):
            # Get traffic snapshot at time t: (num_nodes, 1)
            x = x_seq[t]

            # Spatial processing: GCN extracts features from graph structure
            x = F.relu(self.gc1(x, edge_index))   # (num_nodes, hidden_dim)
            x = F.relu(self.gc2(x, edge_index))   # (num_nodes, hidden_dim)

            # Temporal processing: GRU updates hidden state
            # h carries information from all previous timesteps
            h = self.gru(x, h)   # (num_nodes, hidden_dim)

        # After processing all timesteps, predict the future
        out = self.fc(h)   # (num_nodes, pred_len)
        return out