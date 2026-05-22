"""
generate_data.py — Create realistic synthetic traffic data for the road network.

The data simulates 7 days of traffic at 5-minute intervals (2016 timesteps).
Traffic patterns include:
  - Rush-hour peaks (8-10 AM, 5-7 PM)
  - Night-time lows (11 PM - 5 AM)
  - Slightly lower weekend traffic
  - Road importance based on node degree (more connections = busier)
  - Spatial smoothing with neighbors (nearby roads have correlated traffic)
  - Small Gaussian noise for realism
  - Final normalization to [0, 1]

Usage:
  python data/generate_data.py           # Generates for the small subgraph (training)
  python data/generate_data.py --full    # Generates for the full Pune city (inference)
"""

import numpy as np
import pickle
import networkx as nx
import argparse
import sys
import os

def generate_traffic(graph_path, output_path):
    # ──────────────────────────────────────────────────────────────
    # 1. Load the graph
    # ──────────────────────────────────────────────────────────────
    print(f"Loading graph from {graph_path}...")
    if not os.path.exists(graph_path):
        print(f"Error: {graph_path} not found.")
        sys.exit(1)

    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    # Convert MultiDiGraph to simple undirected for neighbor lookups
    G_undirected = G.to_undirected()

    num_nodes = len(G.nodes)
    node_list = list(G.nodes)
    print(f"Total nodes: {num_nodes}")

    # ──────────────────────────────────────────────────────────────
    # 2. Time configuration
    # ──────────────────────────────────────────────────────────────
    DAYS = 7                    # one full week
    STEPS_PER_DAY = 288         # 24 hours × 60 min / 5-min intervals = 288
    TOTAL_STEPS = DAYS * STEPS_PER_DAY   # 2016
    STEP_MINUTES = 5            # each timestep = 5 minutes

    print(f"Generating {TOTAL_STEPS} timesteps ({DAYS} days × {STEPS_PER_DAY} steps/day)")

    # ──────────────────────────────────────────────────────────────
    # 3. Build the temporal profile for one full week
    # ──────────────────────────────────────────────────────────────
    temporal_profile = np.zeros(TOTAL_STEPS)

    for t in range(TOTAL_STEPS):
        day_index = t // STEPS_PER_DAY          # which day (0 = Mon, ..., 6 = Sun)
        step_in_day = t % STEPS_PER_DAY         # which 5-min slot within the day
        hour = (step_in_day * STEP_MINUTES) / 60.0   # fractional hour (0.0 to 23.99)

        base = 0.3

        # Morning rush hour: 8 AM - 10 AM
        if 8.0 <= hour <= 10.0:
            phase = (hour - 8.0) / 2.0 * np.pi
            base += 0.5 * np.sin(phase)

        # Evening rush hour: 5 PM - 7 PM
        elif 17.0 <= hour <= 19.0:
            phase = (hour - 17.0) / 2.0 * np.pi
            base += 0.45 * np.sin(phase)

        # Midday
        elif 10.0 < hour < 17.0:
            base += 0.15

        # Night-time low
        elif hour >= 23.0 or hour <= 5.0:
            base = 0.05

        # Weekend reduction
        is_weekend = day_index >= 5
        if is_weekend:
            base *= 0.7

        temporal_profile[t] = base

    print("Temporal profile created (rush hours, nights, weekends)")

    # ──────────────────────────────────────────────────────────────
    # 4. Compute node importance from graph degree
    # ──────────────────────────────────────────────────────────────
    degrees = np.array([G_undirected.degree(n) for n in node_list], dtype=np.float32)

    degree_min = degrees.min()
    degree_max = degrees.max()
    if degree_max > degree_min:
        node_importance = (degrees - degree_min) / (degree_max - degree_min)
    else:
        node_importance = np.ones(num_nodes) * 0.5

    node_importance = 0.7 + 0.6 * node_importance

    # ──────────────────────────────────────────────────────────────
    # 5. Generate the full traffic matrix (TOTAL_STEPS, num_nodes)
    # ──────────────────────────────────────────────────────────────
    print("Generating traffic matrix...")
    traffic_data = np.outer(temporal_profile, node_importance)

    # ──────────────────────────────────────────────────────────────
    # 6. Add Gaussian noise
    # ──────────────────────────────────────────────────────────────
    noise = np.random.normal(0, 0.03, size=traffic_data.shape)
    traffic_data += noise

    # ──────────────────────────────────────────────────────────────
    # 7. Fast Spatial smoothing using sparse adjacency matrix
    # ──────────────────────────────────────────────────────────────
    print("Applying fast spatial smoothing...")
    A = nx.to_scipy_sparse_array(G_undirected, nodelist=node_list, format='csr')

    # Transpose to (N, T) for matrix multiplication
    traffic_T = traffic_data.T
    
    # Calculate sum of neighbors: A.dot(traffic_T)
    neighbor_sum = A.dot(traffic_T)
    
    # Calculate degree of each node for averaging
    deg = np.array(A.sum(axis=1)).reshape(-1, 1) # (N, 1)
    deg[deg == 0] = 1   # Avoid division by zero
    
    # Average = sum / degree
    neighbor_avg = neighbor_sum / deg
    
    # Blend: 70% self, 30% neighbors
    SELF_WEIGHT = 0.7
    NEIGHBOR_WEIGHT = 0.3
    smoothed_T = SELF_WEIGHT * traffic_T + NEIGHBOR_WEIGHT * neighbor_avg
    
    # Transpose back to (T, N)
    traffic_data = smoothed_T.T

    # ──────────────────────────────────────────────────────────────
    # 8. Normalize to [0, 1] range
    # ──────────────────────────────────────────────────────────────
    traffic_data = np.clip(traffic_data, 0, None)

    data_min = traffic_data.min()
    data_max = traffic_data.max()
    if data_max > data_min:
        traffic_data = (traffic_data - data_min) / (data_max - data_min)
    else:
        traffic_data = np.zeros_like(traffic_data)

    print(f"Normalized to [0, 1]. Range: {traffic_data.min():.4f} to {traffic_data.max():.4f}")

    # ──────────────────────────────────────────────────────────────
    # 9. Save the dataset
    # ──────────────────────────────────────────────────────────────
    np.save(output_path, traffic_data.astype(np.float32))

    print(f"\nTraffic dataset saved to {output_path}")
    print(f"Shape: {traffic_data.shape}  (timesteps × nodes)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Generate data for the full city graph")
    args = parser.parse_args()

    if args.full:
        graph_file = "graph/graph_data.pkl"
        output_file = "data/pune_traffic_data.npy"
    else:
        graph_file = "graph/subgraph.pkl"
        output_file = "data/traffic_data.npy"

    generate_traffic(graph_file, output_file)