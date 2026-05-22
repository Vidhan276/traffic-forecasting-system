import torch
import numpy as np
import networkx as nx
import osmnx as ox
import folium
import math
from torch_geometric.utils import from_networkx

# Predefined locations in Pune for fast lookup
PUNE_LOCATIONS = {
    'Shivajinagar': (18.5325915, 73.8513115),
    'Viman Nagar': (18.5703877, 73.9133336),
    'Hinjewadi': (18.5920497, 73.7579029),
    'Katraj': (18.4536792, 73.8563196),
    'Koregaon Park': (18.5366225, 73.8932738),
    'Kothrud': (18.5072618, 73.8056676),
    'Hadapsar': (18.5007741, 73.9379146),
    'Baner': (18.5589937, 73.7845232),
    'Swargate': (18.4986771, 73.8578427)
}

# Google Maps traffic colors
TRAFFIC_COLORS = [
    (0.0,  "#34A853"),   # Green — free flow
    (0.25, "#FBBC04"),   # Yellow — light traffic
    (0.50, "#F9AB00"),   # Orange — moderate traffic
    (0.75, "#EA4335"),   # Red — heavy traffic
    (1.0,  "#C5221F"),   # Dark Red — severe congestion
]

def get_traffic_color(value):
    """Map a traffic value (0 to 1) to a Google Maps-style traffic color."""
    value = max(0.0, min(1.0, value))
    for i in range(len(TRAFFIC_COLORS) - 1):
        low_val, low_color = TRAFFIC_COLORS[i]
        high_val, high_color = TRAFFIC_COLORS[i + 1]
        if low_val <= value <= high_val:
            t = (value - low_val) / (high_val - low_val) if high_val > low_val else 0
            r1, g1, b1 = int(low_color[1:3], 16), int(low_color[3:5], 16), int(low_color[5:7], 16)
            r2, g2, b2 = int(high_color[1:3], 16), int(high_color[3:5], 16), int(high_color[5:7], 16)
            r = int(r1 + t * (r2 - r1))
            g = int(g1 + t * (g2 - g1))
            b = int(b1 + t * (b2 - b1))
            return f"#{r:02x}{g:02x}{b:02x}"
    return TRAFFIC_COLORS[-1][1]

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance in meters between two lat/lon points."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def find_nearest_node(G, lat, lon):
    """Find the nearest node in the graph to the given lat/lon."""
    return ox.distance.nearest_nodes(G, lon, lat)

def generate_route_forecast(G, model, scaler, traffic_data, origin_name, dest_name, output_path="route_forecast_map.html"):
    """
    Finds the shortest path between A and B, runs the T-GCN model on the full graph
    to predict traffic 15 mins ahead, and overlays it on a Folium map.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Get origin and destination nodes
    o_lat, o_lon = PUNE_LOCATIONS[origin_name]
    d_lat, d_lon = PUNE_LOCATIONS[dest_name]
    
    origin_node = find_nearest_node(G, o_lat, o_lon)
    dest_node = find_nearest_node(G, d_lat, d_lon)
    
    # 2. Find shortest path using road length
    try:
        route = nx.shortest_path(G, origin_node, dest_node, weight='length')
    except nx.NetworkXNoPath:
        return False, {"error": "No path found between these locations."}
        
    # Calculate route length
    route_length_m = sum(ox.utils_graph.get_route_edge_attributes(G, route, 'length'))
    route_length_km = route_length_m / 1000.0
    
    # 3. Predict traffic for the entire graph
    # We need the last SEQ_LEN (12) timesteps
    seq_len = 12
    last_traffic = traffic_data[-seq_len:] # (12, num_nodes)
    
    # Normalize
    traffic_norm = (last_traffic - scaler["mean"]) / scaler["std"]
    
    # Run model
    x_seq = torch.tensor(traffic_norm, dtype=torch.float32).unsqueeze(-1).to(device)
    
    # Strip attributes for PyG conversion (only if not already done)
    G_clean = G.copy()
    for node in G_clean.nodes:
        G_clean.nodes[node].clear()
    for u, v, k in G_clean.edges(keys=True):
        G_clean.edges[u, v, k].clear()
        
    edge_index = from_networkx(G_clean).edge_index.to(device)
    
    model.eval()
    with torch.no_grad():
        prediction = model(x_seq, edge_index) # (num_nodes, pred_len)
        
    # Un-normalize the 15-minute forecast (first step)
    pred_np = prediction.cpu().numpy()[:, 0]
    predicted_traffic = pred_np * scaler["std"] + scaler["mean"]
    
    # Normalize to [0, 1] for coloring based on min/max of prediction
    t_min, t_max = predicted_traffic.min(), predicted_traffic.max()
    if t_max > t_min:
        norm_values = (predicted_traffic - t_min) / (t_max - t_min)
    else:
        norm_values = np.ones_like(predicted_traffic) * 0.5
        
    # Build dictionary of node_id -> normalized traffic
    node_list = list(G.nodes)
    traffic_dict = {node_id: float(norm_values[i]) for i, node_id in enumerate(node_list)}
    
    # 4. Generate the map
    center_lat = (o_lat + d_lat) / 2
    center_lon = (o_lon + d_lon) / 2
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="cartodbpositron")
    
    # Map title
    title_html = f"""
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
                z-index: 1000; background: white; padding: 8px 16px; border-radius: 6px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.3); font-family: Arial, sans-serif;
                font-size: 14px; font-weight: bold;">
        Route Forecast: {origin_name} ➔ {dest_name} (15 mins ahead)
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Draw route segments
    route_traffic_sum = 0
    route_segments = 0
    
    # We iterate over pairs of nodes in the route
    for i in range(len(route) - 1):
        u = route[i]
        v = route[i+1]
        
        # Get edge data
        edge_data = min(G.get_edge_data(u, v).values(), key=lambda x: x.get('length', float('inf')))
        
        # Traffic is average of the two nodes
        val = (traffic_dict.get(u, 0.3) + traffic_dict.get(v, 0.3)) / 2
        
        # Road type weighting
        highway = edge_data.get("highway", "residential")
        if isinstance(highway, list): highway = highway[0]
            
        if highway in ["motorway", "trunk"]: val += 0.25; weight = 8
        elif highway in ["primary"]: val += 0.20; weight = 7
        elif highway in ["secondary"]: val += 0.10; weight = 6
        elif highway in ["tertiary"]: val -= 0.05; weight = 5
        else: val -= 0.15; weight = 4
            
        val = max(0.0, min(1.0, val))
        color = get_traffic_color(val)
        
        route_traffic_sum += val
        route_segments += 1
        
        # Geometry
        if 'geometry' in edge_data:
            coords = [(y, x) for x, y in edge_data['geometry'].coords]
        else:
            coords = [(G.nodes[u]['y'], G.nodes[u]['x']), (G.nodes[v]['y'], G.nodes[v]['x'])]
            
        road_name = edge_data.get('name', 'Unnamed Road')
        if isinstance(road_name, list): road_name = road_name[0]
            
        folium.PolyLine(
            coords, color=color, weight=weight, opacity=0.9,
            tooltip=f"{road_name} ({highway})"
        ).add_to(m)
        
    # Add markers for start and end
    folium.Marker([o_lat, o_lon], tooltip="Origin: " + origin_name, icon=folium.Icon(color='green', icon='play')).add_to(m)
    folium.Marker([d_lat, d_lon], tooltip="Destination: " + dest_name, icon=folium.Icon(color='red', icon='stop')).add_to(m)
    
    m.save(output_path)
    
    # Calculate summary
    avg_traffic = route_traffic_sum / route_segments if route_segments > 0 else 0
    
    # Base time: assuming 40 km/h average
    base_time_mins = (route_length_km / 40.0) * 60
    # Penalty based on traffic (up to 2.5x slower)
    delay_multiplier = 1.0 + (avg_traffic * 1.5)
    est_time_mins = base_time_mins * delay_multiplier
    
    summary = {
        "distance_km": round(route_length_km, 2),
        "base_time_mins": round(base_time_mins),
        "est_time_mins": round(est_time_mins),
        "avg_traffic_level": avg_traffic
    }
    
    return True, summary
