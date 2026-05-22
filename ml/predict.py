"""
predict.py — Generate traffic prediction maps for the Kothrud area.

This script:
  1. Loads the Kothrud road network graph
  2. Loads the trained T-GCN model
  3. Uses the last hour of data to predict traffic 15 minutes ahead
  4. Creates TWO interactive Folium maps:
     - pune_current_traffic.html  (last known traffic state)
     - pune_predicted_traffic.html (predicted 15-min-ahead traffic)

The model generalizes to different-sized graphs because GCN/GRU dimensions
are independent of the number of nodes — they process per-node features.

Run from project root:  python ml/predict.py
"""

import sys
import torch
import pickle
import numpy as np
import osmnx as ox
import folium
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset import load_scaler
from data.loader import load_config, graph_to_edge_index
from models.factory import build_model

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
cfg = load_config()
SEQ_LEN    = cfg["model"]["seq_len"]
PRED_LEN   = cfg["model"]["pred_len"]
HIDDEN_DIM = cfg["model"]["hidden_dim"]

# Google Maps traffic colors (green=free, red=jam)
TRAFFIC_COLORS = [
    (0.0,  "#34A853"),   # Green — free flow
    (0.25, "#FBBC04"),   # Yellow — light traffic
    (0.50, "#F9AB00"),   # Orange — moderate traffic
    (0.75, "#EA4335"),   # Red — heavy traffic
    (1.0,  "#C5221F"),   # Dark Red — severe congestion
]

# ──────────────────────────────────────────────────────────────
# 1. Load the Kothrud graph
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("T-GCN Traffic Forecasting — Pune City Prediction")
print("=" * 60)

print("\nLoading full Pune graph...")
with open("graph/graph_data.pkl", "rb") as f:
    G = pickle.load(f)

num_nodes = len(G.nodes)
print(f"Pune graph: {num_nodes} nodes, {len(G.edges)} edges")

# Prepare edge_index from graph
edge_index = graph_to_edge_index(G)

# ──────────────────────────────────────────────────────────────
# 2. Load traffic data and apply scaler
# ──────────────────────────────────────────────────────────────
print("Loading traffic data and scaler...")
try:
    # mmap_mode avoids loading the entire 1.1GB into RAM just to get the last 12 timesteps
    traffic_data_mmap = np.load("data/pune_traffic_data.npy", mmap_mode='r')
    traffic_input = np.array(traffic_data_mmap[-SEQ_LEN:])
except FileNotFoundError:
    print("Full city data not found. Falling back to subgraph data.")
    traffic_data = np.load("data/traffic_data.npy")
    traffic_input = traffic_data[-SEQ_LEN:]

scaler = load_scaler("ml/scaler.pkl")

# Normalize using training scaler
traffic_norm = (traffic_input - scaler["mean"]) / scaler["std"]

# Keep un-normalized version for "current traffic" map
current_traffic_raw = traffic_input[-1]   # last known timestep, un-normalized

# ──────────────────────────────────────────────────────────────
# 3. Load the trained model and predict
# ──────────────────────────────────────────────────────────────
print("Loading trained model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_model(cfg["model"]).to(device)
model.load_state_dict(torch.load("ml/traffic_model.pth", map_location=device, weights_only=True))
model.eval()

# Prepare input tensor: (seq_len, num_nodes, 1)
x_seq = torch.tensor(traffic_norm, dtype=torch.float32).unsqueeze(-1).to(device)
edge_index = edge_index.to(device)

print("Running prediction...")
with torch.no_grad():
    prediction = model(x_seq, edge_index)   # (num_nodes, pred_len)

# Un-normalize predictions
pred_np = prediction.cpu().numpy()
pred_unnorm = pred_np * scaler["std"] + scaler["mean"]

# We use the first predicted step (5 minutes ahead) for the map
predicted_traffic = pred_unnorm[:, 0]   # (num_nodes,)

print(f"Prediction done. Range: [{predicted_traffic.min():.3f}, {predicted_traffic.max():.3f}]")

# ──────────────────────────────────────────────────────────────
# 4. Helper functions for map generation
# ──────────────────────────────────────────────────────────────
def get_traffic_color(value):
    """
    Map a traffic value (0 to 1) to a Google Maps-style traffic color.
    Interpolates between the defined color stops.
    """
    # Clamp to [0, 1]
    value = max(0.0, min(1.0, value))

    # Find the two color stops that bracket this value
    for i in range(len(TRAFFIC_COLORS) - 1):
        low_val, low_color = TRAFFIC_COLORS[i]
        high_val, high_color = TRAFFIC_COLORS[i + 1]

        if low_val <= value <= high_val:
            # How far between the two stops (0 to 1)
            t = (value - low_val) / (high_val - low_val) if high_val > low_val else 0

            # Interpolate RGB hex colors
            r1, g1, b1 = int(low_color[1:3], 16), int(low_color[3:5], 16), int(low_color[5:7], 16)
            r2, g2, b2 = int(high_color[1:3], 16), int(high_color[3:5], 16), int(high_color[5:7], 16)

            r = int(r1 + t * (r2 - r1))
            g = int(g1 + t * (g2 - g1))
            b = int(b1 + t * (b2 - b1))

            return f"#{r:02x}{g:02x}{b:02x}"

    return TRAFFIC_COLORS[-1][1]   # fallback: darkest red


def add_legend(m):
    """Add a traffic color legend to the Folium map."""
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                background: white; padding: 12px 16px; border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.3); font-family: Arial, sans-serif;">
        <b style="font-size: 13px;">Traffic Level</b><br>
        <div style="margin-top: 6px;">
            <span style="background:#34A853; width:18px; height:12px; display:inline-block; border-radius:2px;"></span>
            <span style="font-size:11px;"> Free Flow</span><br>
            <span style="background:#FBBC04; width:18px; height:12px; display:inline-block; border-radius:2px;"></span>
            <span style="font-size:11px;"> Light Traffic</span><br>
            <span style="background:#F9AB00; width:18px; height:12px; display:inline-block; border-radius:2px;"></span>
            <span style="font-size:11px;"> Moderate</span><br>
            <span style="background:#EA4335; width:18px; height:12px; display:inline-block; border-radius:2px;"></span>
            <span style="font-size:11px;"> Heavy Traffic</span><br>
            <span style="background:#C5221F; width:18px; height:12px; display:inline-block; border-radius:2px;"></span>
            <span style="font-size:11px;"> Severe Congestion</span>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


def create_traffic_map(G, node_list, traffic_values, title, output_path):
    """
    Create an interactive Folium traffic map.

    Args:
        G              : OSMnx graph (with geometry)
        node_list      : list of node IDs
        traffic_values : numpy array of traffic per node (0 to 1 scale)
        title          : map title string
        output_path    : where to save the HTML file
    """
    # Normalize traffic values to [0, 1] for coloring
    t_min, t_max = traffic_values.min(), traffic_values.max()
    if t_max > t_min:
        norm_values = (traffic_values - t_min) / (t_max - t_min)
    else:
        norm_values = np.ones_like(traffic_values) * 0.5

    # Build a lookup: node_id -> normalized traffic value
    traffic_dict = {}
    for i, node_id in enumerate(node_list):
        if i < len(norm_values):
            traffic_dict[node_id] = float(norm_values[i])
        else:
            traffic_dict[node_id] = 0.2   # default for extra nodes

    # Get node/edge GeoDataFrames for coordinates
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)

    # Center of the map
    center_lat = nodes_gdf["y"].mean()
    center_lon = nodes_gdf["x"].mean()

    # Create the Folium map with clean CartoDB Positron tiles
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=15,
        tiles="cartodbpositron"
    )

    # Add a title
    title_html = f"""
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
                z-index: 1000; background: white; padding: 8px 16px; border-radius: 6px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.3); font-family: Arial, sans-serif;
                font-size: 14px; font-weight: bold;">
        {title}
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    if len(edges_gdf) > 20000:
        # Rendering 320,000 edges will crash the browser. Filter to major roads only.
        print(f"  Graph has {len(edges_gdf)} edges. Filtering to major roads for rendering...")
        major = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary", "secondary_link", "tertiary", "tertiary_link"}
        def is_major(h):
            if isinstance(h, list):
                return any(x in major for x in h)
            return h in major
        
        edges_gdf = edges_gdf[edges_gdf["highway"].apply(is_major)]
        print(f"  Filtered down to {len(edges_gdf)} edges.")

    # Draw each road segment colored by traffic
    for _, row in edges_gdf.reset_index().iterrows():
        u = row["u"]
        v = row["v"]

        # Average traffic of the two endpoint nodes
        val = (traffic_dict.get(u, 0.3) + traffic_dict.get(v, 0.3)) / 2

        # --- Road type weighting (same logic as original) ---
        highway = row.get("highway", "residential")
        if isinstance(highway, list):
            highway = highway[0]

        if highway in ["motorway", "trunk"]:
            val += 0.25
            weight = 8
        elif highway in ["primary"]:
            val += 0.20
            weight = 7
        elif highway in ["secondary"]:
            val += 0.10
            weight = 6
        elif highway in ["tertiary"]:
            val -= 0.05
            weight = 5
        else:   # residential / service / shortcuts
            val -= 0.15
            weight = 3

        val = max(0.0, min(1.0, val))    # clamp to [0, 1]

        # Get the color
        color = get_traffic_color(val)

        # Road coordinates (lat, lon pairs for Folium)
        coords = [(y, x) for x, y in row["geometry"].coords]

        # Build tooltip text with road name if available
        road_name = row.get("name", None)
        if isinstance(road_name, list):
            road_name = road_name[0]
        if road_name and str(road_name) != "nan":
            tooltip_text = f"{road_name} ({highway})"
        else:
            tooltip_text = f"{highway} road"

        folium.PolyLine(
            coords,
            color=color,
            weight=weight,
            opacity=0.9,
            tooltip=tooltip_text
        ).add_to(m)

    # Add the color legend
    add_legend(m)

    # Save
    m.save(output_path)
    print(f"  Map saved: {output_path}")

# ──────────────────────────────────────────────────────────────
# 5. Generate the two maps
# ──────────────────────────────────────────────────────────────
node_list = list(G.nodes)

print("\nCreating traffic maps...")

# Map 1: Current traffic (last known state)
create_traffic_map(
    G, node_list, current_traffic_raw,
    title="🚗 Pune Traffic — Current State",
    output_path="pune_current_traffic.html"
)

# Map 2: Predicted traffic (15 minutes ahead)
create_traffic_map(
    G, node_list, predicted_traffic,
    title="🔮 Pune Traffic — Predicted (15 min ahead)",
    output_path="pune_predicted_traffic.html"
)

print("\n" + "=" * 60)
print("PREDICTION COMPLETE")
print("=" * 60)
print("  Current traffic map:   pune_current_traffic.html")
print("  Predicted traffic map: pune_predicted_traffic.html")
print("=" * 60)