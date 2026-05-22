"""POST /predict_nodes — run the GNN and return per-node traffic forecasts."""

import logging
from datetime import datetime, timezone

import numpy as np
import folium
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.concurrency import run_in_threadpool

from api.schemas import PredictNodesRequest, PredictNodesResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Forecasting"])


@router.post(
    "/predict_nodes",
    response_model=PredictNodesResponse,
    summary="Forecast traffic for all nodes",
)
async def predict_nodes(
    body: PredictNodesRequest,
    request: Request,
) -> PredictNodesResponse:
    """
    Run the T-GCN model and return per-node traffic forecasts.

    - If ``traffic_seq`` is provided, uses that as input (shape: seq_len × num_nodes).
    - If omitted, the service reads the latest traffic data from disk automatically.

    Returns forecasts for the next ``horizon_steps`` × 5-minute intervals.
    """
    svc = request.app.state.forecast_svc

    try:
        if body.traffic_seq is not None:
            seq = np.array(body.traffic_seq, dtype=np.float32)
            if seq.ndim != 2:
                raise HTTPException(
                    status_code=422,
                    detail=f"traffic_seq must be 2-D (seq_len × num_nodes), got shape {seq.shape}",
                )
            pred_unnorm = svc.predict(traffic_seq=seq)
        else:
            pred_unnorm = svc.predict()

        # pred_unnorm: (num_nodes, pred_len)
        # Return only the requested number of steps
        steps = min(body.horizon_steps, pred_unnorm.shape[1])
        pred_slice = pred_unnorm[:, :steps]  # (num_nodes, steps)

        # Transpose to (steps, num_nodes) for JSON serialisation
        forecasts = pred_slice.T.tolist()

        return PredictNodesResponse(
            num_nodes=pred_slice.shape[0],
            horizon_steps=steps,
            forecasts=forecasts,
            computed_at=datetime.now(timezone.utc).isoformat(),
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Error in /predict_nodes")
        raise HTTPException(status_code=500, detail=str(e))


# ── Optimized Map Generation served directly from Backend ────────────────────

def _generate_backend_map(app, is_predicted: bool) -> str:
    forecast_svc = app.state.forecast_svc
    routing_svc = app.state.routing_svc
    
    if forecast_svc is None or routing_svc is None:
        raise HTTPException(status_code=503, detail="Services not initialized")
        
    G = routing_svc.G
    node_list = forecast_svc.node_list
    
    if is_predicted:
        pred_unnorm = forecast_svc.predict()
        traffic_values = pred_unnorm[:, 0]
        title = "🔮 Pune Traffic — Predicted (15 min ahead)"
    else:
        traffic_values = forecast_svc.get_current_traffic()
        title = "🚗 Pune Traffic — Current State"
        
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
            traffic_dict[node_id] = 0.2

    # Center of the map calculated directly from NetworkX nodes to save memory
    lats = [d["y"] for n, d in G.nodes(data=True) if "y" in d]
    lons = [d["x"] for n, d in G.nodes(data=True) if "x" in d]
    center_lat = sum(lats) / len(lats) if lats else 18.52
    center_lon = sum(lons) / len(lons) if lons else 73.855

    # Create the Folium map
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

    # Google Maps traffic colors
    TRAFFIC_COLORS = [
        (0.0,  "#34A853"),   # Green
        (0.25, "#FBBC04"),   # Yellow
        (0.50, "#F9AB00"),   # Orange
        (0.75, "#EA4335"),   # Red
        (1.0,  "#C5221F"),   # Dark Red
    ]

    def get_traffic_color(value):
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

    # Group coordinates by (color, weight) to minimize Leaflet DOM output size from 18MB to 5MB
    group_map = {}

    # Iterate directly over NetworkX edges to bypass heavy OSMnx DataFrame conversion
    for u, v, data in G.edges(data=True):
        val = (traffic_dict.get(u, 0.3) + traffic_dict.get(v, 0.3)) / 2

        highway = data.get("highway", "residential")
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
        else:
            val -= 0.15
            weight = 3

        val = max(0.0, min(1.0, val))
        color = get_traffic_color(val)
        
        # Get coordinates from edge geometry or fallback to straight line
        geom = data.get("geometry")
        if geom:
            coords = [(y, x) for x, y in geom.coords]
        else:
            coords = [
                (G.nodes[u]["y"], G.nodes[u]["x"]),
                (G.nodes[v]["y"], G.nodes[v]["x"])
            ]
        
        group_key = (color, weight)
        if group_key not in group_map:
            group_map[group_key] = []
        group_map[group_key].append(coords)

    # Add grouped multi-polylines to the map
    for (color, weight), coords_list in group_map.items():
        if coords_list:
            folium.PolyLine(
                coords_list,
                color=color,
                weight=weight,
                opacity=0.85
            ).add_to(m)

    # Add the legend
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

    return m._repr_html_()


@router.get("/maps/current", response_class=HTMLResponse)
async def get_current_map(request: Request):
    if not hasattr(request.app.state, "current_map_html") or request.app.state.current_map_html is None:
        html = await run_in_threadpool(_generate_backend_map, request.app, False)
        request.app.state.current_map_html = html
    return HTMLResponse(content=request.app.state.current_map_html)


@router.get("/maps/predicted", response_class=HTMLResponse)
async def get_predicted_map(request: Request):
    if not hasattr(request.app.state, "predicted_map_html") or request.app.state.predicted_map_html is None:
        html = await run_in_threadpool(_generate_backend_map, request.app, True)
        request.app.state.predicted_map_html = html
    return HTMLResponse(content=request.app.state.predicted_map_html)
