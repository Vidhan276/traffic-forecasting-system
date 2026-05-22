"""
routing.py — RoutingService

Future-aware multi-route routing over the Pune road network.

Key design:
  1. Uses GNN-predicted node traffic values to derive edge travel-time costs.
  2. Runs Dijkstra three times with three different cost functions to produce
     route variants:
       - "Fastest Now"    : minimise current travel time.
       - "Fastest 15-min" : minimise predicted travel time (GNN forecast).
       - "Balanced"       : minimise 0.5 × distance + 0.5 × predicted congestion.
  3. Graph and static edge attributes (length, geometry, road type) are cached
     as instance attributes and never recomputed between requests.

Edge cost formula:
  speed_factor = 1 - JAM_FACTOR × avg_traffic   (avg_traffic ∈ [0, 1])
  speed_kmh    = FREE_FLOW_KMH × speed_factor
  cost_sec     = edge_length_m / (speed_kmh / 3.6)

Traffic colour mapping (Google Maps style):
  [0.00, 0.25) → Green  #34A853  (free flow)
  [0.25, 0.50) → Yellow #FBBC04  (light)
  [0.50, 0.75) → Orange #F9AB00  (moderate)
  [0.75, 1.00] → Red    #EA4335  (heavy → dark red #C5221F at 1.0)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
import osmnx as ox
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

# ── Traffic colour stops (value, hex) ─────────────────────────────────────────
_TRAFFIC_COLORS: list[tuple[float, str]] = [
    (0.00, "#34A853"),
    (0.25, "#FBBC04"),
    (0.50, "#F9AB00"),
    (0.75, "#EA4335"),
    (1.00, "#C5221F"),
]


def _get_color(value: float) -> str:
    """Interpolate a hex colour from the Google-Maps-style traffic scale."""
    value = max(0.0, min(1.0, float(value)))
    for i in range(len(_TRAFFIC_COLORS) - 1):
        lo_v, lo_c = _TRAFFIC_COLORS[i]
        hi_v, hi_c = _TRAFFIC_COLORS[i + 1]
        if lo_v <= value <= hi_v:
            t  = (value - lo_v) / (hi_v - lo_v) if hi_v > lo_v else 0.0
            r1, g1, b1 = int(lo_c[1:3], 16), int(lo_c[3:5], 16), int(lo_c[5:7], 16)
            r2, g2, b2 = int(hi_c[1:3], 16), int(hi_c[3:5], 16), int(hi_c[5:7], 16)
            r = int(r1 + t * (r2 - r1))
            g = int(g1 + t * (g2 - g1))
            b = int(b1 + t * (b2 - b1))
            return f"#{r:02x}{g:02x}{b:02x}"
    return _TRAFFIC_COLORS[-1][1]


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class SegmentInfo:
    """Per-edge display info for a map polyline."""
    coords:     list[tuple[float, float]]  # [(lat, lon), ...]
    color:      str
    congestion: float                      # 0..1
    road_name:  str
    highway:    str


@dataclass
class RouteResult:
    """A single route option returned by the routing service."""
    label:             str              # "Fastest Now" / "Fastest 15-min" / "Balanced"
    color:             str              # route line colour on the overview map
    nodes:             list[int]        # OSM node IDs along the route
    distance_km:       float
    eta_now_min:       float            # travel time with current traffic (minutes)
    eta_future_min:    float            # travel time with 15-min forecast (minutes)
    congestion_score:  float            # mean predicted traffic along route [0..1]
    segments:          list[SegmentInfo] = field(default_factory=list)


# ── Service ────────────────────────────────────────────────────────────────────

class RoutingService:
    """
    Future-aware multi-route routing over the Pune road network.

    Args:
        G          : NetworkX MultiDiGraph from OSMnx (with geometry attrs).
        node_list  : ordered list of node IDs matching the traffic data columns.
        cfg        : full config dict (reads routing.* sub-section).
    """

    # Route display colours on the map
    _ROUTE_COLORS = {
        "Fastest Now":    "#1A73E8",   # Google blue
        "Fastest 15-min": "#34A853",   # Google green
        "Balanced":       "#F9AB00",   # Google orange
    }

    def __init__(
        self,
        G,
        node_list: list[int],
        cfg: dict[str, Any],
    ):
        self.G         = G
        self.node_list = node_list
        self.node_idx  = {node: i for i, node in enumerate(node_list)}

        routing_cfg = cfg.get("routing", {})
        self._free_flow_kmh = routing_cfg.get("free_flow_kmh", 30.0)
        self._jam_factor    = routing_cfg.get("jam_factor", 0.7)
        self._bal_dist_w    = routing_cfg.get("balanced_dist_weight", 0.5)
        self._bal_cong_w    = routing_cfg.get("balanced_cong_weight", 0.5)
        self._locations     = routing_cfg.get("locations", {})

        logger.info("RoutingService: mapping road nodes to nearest forecast nodes...")
        self._forecast_idx_by_node = self._build_forecast_node_lookup()

        # Precache edge attributes for vectorized numpy computation
        logger.info("RoutingService: caching edge endpoints and lengths for vectorized execution...")
        u_idx_list = []
        v_idx_list = []
        lengths_list = []
        edge_refs = []
        self._edge_uv = []

        for u, v, k, data in self.G.edges(keys=True, data=True):
            u_idx_list.append(self._forecast_idx_by_node[u])
            v_idx_list.append(self._forecast_idx_by_node[v])
            lengths_list.append(float(data.get("length", 50.0)))
            edge_refs.append((u, v, k, data))
            self._edge_uv.append((u, v))
            
        self._u_idx_arr = np.array(u_idx_list, dtype=np.int32)
        self._v_idx_arr = np.array(v_idx_list, dtype=np.int32)
        self._lengths_arr = np.array(lengths_list, dtype=np.float64)
        self._edge_refs = edge_refs
        self._nearest_route_node = self._build_location_node_lookup()
        self._path_cache: dict[tuple[str, str, int], tuple[float, list[RouteResult]]] = {}
        self._path_cache_ttl_sec = float(routing_cfg.get("route_cache_ttl_sec", 120.0))

        logger.info("RoutingService ready: %d forecasting nodes, %d routing edges", len(node_list), G.number_of_edges())

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_routes(
        self,
        origin_name: str,
        dest_name: str,
        current_traffic: np.ndarray,
        predicted_traffic: np.ndarray,
    ) -> list[RouteResult]:
        """
        Compute three route variants between origin and destination.

        Args:
            origin_name       : key in config.routing.locations.
            dest_name         : key in config.routing.locations.
            current_traffic   : (num_nodes,) array of current traffic [0..1].
            predicted_traffic : (num_nodes,) array of GNN-predicted traffic [0..1].

        Returns:
            List of up to 3 RouteResult objects.
        """
        if origin_name not in self._locations:
            raise ValueError(f"Unknown origin '{origin_name}'. Available: {list(self._locations.keys())}")
        if dest_name not in self._locations:
            raise ValueError(f"Unknown destination '{dest_name}'. Available: {list(self._locations.keys())}")
        if origin_name == dest_name:
            raise ValueError("Origin and destination must be different.")

        traffic_signature = (
            int(np.round(float(np.mean(current_traffic)) * 10000)),
            int(np.round(float(np.mean(predicted_traffic)) * 10000)),
            int(np.round(float(np.max(predicted_traffic)) * 10000)),
        )
        cache_key = (origin_name, dest_name, hash(traffic_signature))
        cached = self._path_cache.get(cache_key)
        if cached and time.monotonic() - cached[0] < self._path_cache_ttl_sec:
            return cached[1]

        origin_node = self._nearest_route_node[origin_name]
        dest_node = self._nearest_route_node[dest_name]

        self._assign_dynamic_costs(current_traffic, predicted_traffic)

        results: list[RouteResult] = []

        variants = [
            ("Fastest Now",    "cost_now",      current_traffic,   predicted_traffic),
            ("Fastest 15-min", "cost_future",   current_traffic,   predicted_traffic),
            ("Balanced",       "cost_balanced", current_traffic,   predicted_traffic),
        ]

        seen_routes: set[tuple[int, ...]] = set()  # deduplicate identical paths

        for label, weight_attr, cur_t, pred_t in variants:
            try:
                path = nx.astar_path(
                    self.G,
                    origin_node,
                    dest_node,
                    heuristic=self._travel_time_heuristic,
                    weight=weight_attr,
                )
            except nx.NetworkXNoPath:
                logger.warning("No path found for %s -> %s (%s)", origin_name, dest_name, label)
                continue

            path_key = tuple(path)
            if path_key in seen_routes:
                continue
            seen_routes.add(path_key)

            result = self._build_route_result(label, path, cur_t, pred_t)
            results.append(result)

        self._path_cache[cache_key] = (time.monotonic(), results)
        return results

    def location_names(self) -> list[str]:
        """Return the list of named locations available for routing."""
        return list(self._locations.keys())

    def _build_forecast_node_lookup(self) -> dict[int, int]:
        """Map every routing graph node to the nearest available forecast node."""
        forecast_nodes = [node for node in self.node_list if node in self.G.nodes]
        if not forecast_nodes:
            raise ValueError("None of the forecast nodes are present in the routing graph.")

        forecast_coords = np.array(
            [(self.G.nodes[node]["y"], self.G.nodes[node]["x"]) for node in forecast_nodes],
            dtype=np.float64,
        )
        forecast_indices = np.array([self.node_idx[node] for node in forecast_nodes], dtype=np.int32)
        tree = cKDTree(forecast_coords)

        route_nodes = list(self.G.nodes)
        route_coords = np.array(
            [(self.G.nodes[node]["y"], self.G.nodes[node]["x"]) for node in route_nodes],
            dtype=np.float64,
        )
        _, nearest = tree.query(route_coords, k=1)

        return {
            node: int(forecast_indices[nearest_pos])
            for node, nearest_pos in zip(route_nodes, nearest)
        }

    def _build_location_node_lookup(self) -> dict[str, int]:
        return {
            name: ox.distance.nearest_nodes(self.G, lon, lat)
            for name, (lat, lon) in self._locations.items()
        }

    def _travel_time_heuristic(self, u: int, v: int) -> float:
        u_node = self.G.nodes[u]
        v_node = self.G.nodes[v]
        lat1 = math.radians(float(u_node["y"]))
        lat2 = math.radians(float(v_node["y"]))
        d_lat = lat2 - lat1
        d_lon = math.radians(float(v_node["x"]) - float(u_node["x"]))
        a = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2
        distance_m = 6371000.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return distance_m / (self._free_flow_kmh / 3.6)

    def _assign_dynamic_costs(
        self,
        current_traffic: np.ndarray,
        predicted_traffic: np.ndarray,
    ) -> None:
        now_costs = self._edge_time_costs(current_traffic)
        future_costs = self._edge_time_costs(predicted_traffic)
        balanced_costs = self._bal_dist_w * self._lengths_arr + self._bal_cong_w * future_costs

        for i, (_, _, _, data) in enumerate(self._edge_refs):
            data["cost_now"] = float(now_costs[i])
            data["cost_future"] = float(future_costs[i])
            data["cost_balanced"] = float(balanced_costs[i])

    def _edge_time_costs(self, node_traffic: np.ndarray) -> np.ndarray:
        t_ext = np.clip(node_traffic, 0.0, 1.0)
        avg_t = (t_ext[self._u_idx_arr] + t_ext[self._v_idx_arr]) / 2.0
        speed_factor = np.maximum(0.05, 1.0 - self._jam_factor * avg_t)
        speed_ms = (self._free_flow_kmh / 3.6) * speed_factor
        return self._lengths_arr / speed_ms

    # ── Private: edge weight computation ──────────────────────────────────────

    def _compute_edge_weights(
        self,
        node_traffic: np.ndarray,
        cost_type: str = "time",
    ) -> dict[tuple[int, int], float]:
        """
        Compute per-edge cost weights from node traffic values using vectorized numpy.

        Edge cost:
          avg_traffic  = (traffic[u] + traffic[v]) / 2
          speed_factor = 1 - jam_factor × avg_traffic
          speed_kmh    = free_flow_kmh × speed_factor
          time_cost    = edge_length_m / (speed_kmh / 3.6)   [seconds]

        For "balanced", cost = 0.5 × time_cost + 0.5 × length_m.

        Returns:
            Dict mapping (u, v) -> minimum cost across parallel edges
        """
        t_ext = np.clip(node_traffic, 0.0, 1.0)

        # Vectorized lookups
        u_t = t_ext[self._u_idx_arr]
        v_t = t_ext[self._v_idx_arr]
        avg_t = (u_t + v_t) / 2.0

        # Speed and travel time calculations
        speed_factor = np.maximum(0.05, 1.0 - self._jam_factor * avg_t)
        speed_ms     = (self._free_flow_kmh / 3.6) * speed_factor
        time_cost    = self._lengths_arr / speed_ms

        if cost_type == "time":
            costs = time_cost
        elif cost_type == "balanced":
            costs = self._bal_dist_w * self._lengths_arr + self._bal_cong_w * time_cost
        else:
            costs = time_cost

        # Group by (u, v) to find minimum cost across parallel edges
        min_edge_cost = {}
        for uv, cost in zip(self._edge_uv, costs):
            if uv not in min_edge_cost or cost < min_edge_cost[uv]:
                min_edge_cost[uv] = cost

        return min_edge_cost

    # ── Private: route building ────────────────────────────────────────────────

    def _build_route_result(
        self,
        label: str,
        path: list[int],
        current_traffic: np.ndarray,
        predicted_traffic: np.ndarray,
    ) -> RouteResult:
        """Assemble a RouteResult from a node path."""
        dist_m        = 0.0
        eta_now_sec   = 0.0
        eta_fut_sec   = 0.0
        route_cong    = []
        segments: list[SegmentInfo] = []

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]

            # Pick the shortest parallel edge (min length)
            edge_data = min(
                self.G.get_edge_data(u, v).values(),
                key=lambda d: d.get("length", 9999),
            )
            length_m = float(edge_data.get("length", 50.0))
            dist_m  += length_m

            highway = edge_data.get("highway", "residential")
            if isinstance(highway, list):
                highway = highway[0]

            road_name = edge_data.get("name", "Unnamed Road")
            if isinstance(road_name, list):
                road_name = road_name[0]
            if not road_name or str(road_name) == "nan":
                road_name = f"{highway} road"

            # Current and future traffic for this edge
            u_idx = self._forecast_idx_by_node[u]
            v_idx = self._forecast_idx_by_node[v]
            cur_t = (current_traffic[u_idx] + current_traffic[v_idx]) / 2
            pred_t = (predicted_traffic[u_idx] + predicted_traffic[v_idx]) / 2

            route_cong.append(float(pred_t))

            # ETA computations
            def _time(t, l):
                sf = max(0.05, 1.0 - self._jam_factor * t)
                return l / ((self._free_flow_kmh / 3.6) * sf)

            eta_now_sec += _time(cur_t, length_m)
            eta_fut_sec += _time(pred_t, length_m)

            # Geometry: guard against None value (may be absent or explicitly None)
            geom = edge_data.get("geometry", None)
            if geom is not None:
                coords = [(y, x) for x, y in geom.coords]
            else:
                coords = [
                    (self.G.nodes[u]["y"], self.G.nodes[u]["x"]),
                    (self.G.nodes[v]["y"], self.G.nodes[v]["x"]),
                ]

            segments.append(SegmentInfo(
                coords=coords,
                color=_get_color(pred_t),
                congestion=round(float(pred_t), 3),
                road_name=road_name,
                highway=highway,
            ))

        return RouteResult(
            label=label,
            color=self._ROUTE_COLORS.get(label, "#1A73E8"),
            nodes=path,
            distance_km=round(float(dist_m) / 1000, 2),
            eta_now_min=round(float(eta_now_sec) / 60, 1),
            eta_future_min=round(float(eta_fut_sec) / 60, 1),
            congestion_score=round(float(np.mean(route_cong)) if route_cong else 0.3, 3),
            segments=segments,
        )
