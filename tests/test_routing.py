"""
test_routing.py — Unit tests for the routing service logic.

Tests the edge cost computation and route result structure without
requiring a real OSMnx graph — uses a small synthetic NetworkX graph.
"""

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from services.routing import RoutingService, _get_color, RouteResult


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_small_graph():
    """
    Create a tiny synthetic road graph with real-like attributes.
    5 nodes arranged in a line with a shortcut.
    """
    G = nx.MultiDiGraph()

    nodes = {
        0: {"y": 18.50, "x": 73.80},
        1: {"y": 18.51, "x": 73.81},
        2: {"y": 18.52, "x": 73.82},
        3: {"y": 18.53, "x": 73.83},
        4: {"y": 18.54, "x": 73.84},
    }
    for n, attrs in nodes.items():
        G.add_node(n, **attrs)

    edges = [
        (0, 1, {"length": 500, "highway": "primary",     "name": "Main Rd",  "geometry": None}),
        (1, 2, {"length": 600, "highway": "primary",     "name": "Main Rd",  "geometry": None}),
        (2, 3, {"length": 400, "highway": "secondary",   "name": "Cross Rd", "geometry": None}),
        (3, 4, {"length": 500, "highway": "residential", "name": "Side Ln",  "geometry": None}),
        (0, 2, {"length": 900, "highway": "trunk",       "name": "Bypass",   "geometry": None}),
        # Reverse edges for bidirectional routing
        (1, 0, {"length": 500, "highway": "primary",     "name": "Main Rd",  "geometry": None}),
        (2, 1, {"length": 600, "highway": "primary",     "name": "Main Rd",  "geometry": None}),
        (3, 2, {"length": 400, "highway": "secondary",   "name": "Cross Rd", "geometry": None}),
        (4, 3, {"length": 500, "highway": "residential", "name": "Side Ln",  "geometry": None}),
        (2, 0, {"length": 900, "highway": "trunk",       "name": "Bypass",   "geometry": None}),
    ]
    for u, v, data in edges:
        G.add_edge(u, v, **data)

    return G


def _make_cfg():
    return {
        "routing": {
            "free_flow_kmh": 30.0,
            "jam_factor": 0.7,
            "balanced_dist_weight": 0.5,
            "balanced_cong_weight": 0.5,
            "locations": {
                "NodeA": [18.50, 73.80],
                "NodeD": [18.53, 73.83],
            },
        }
    }


# ── Monkeypatch osmnx ──────────────────────────────────────────────────────────
# The routing service calls ox.graph_to_gdfs and ox.distance.nearest_nodes.
# We mock these to avoid needing a real OSMnx graph.

import geopandas as gpd
from shapely.geometry import LineString, Point
import pandas as pd


def _mock_graph_to_gdfs(G):
    nodes = [{"osmid": n, "x": d["x"], "y": d["y"],
              "geometry": Point(d["x"], d["y"])} for n, d in G.nodes(data=True)]
    nodes_gdf = gpd.GeoDataFrame(nodes, geometry="geometry").set_index("osmid")

    edges = []
    for u, v, k, d in G.edges(keys=True, data=True):
        nx_d = d.copy()
        nx_d.update({"u": u, "v": v, "key": k})
        if nx_d.get("geometry") is None:
            nx_d["geometry"] = LineString([
                (G.nodes[u]["x"], G.nodes[u]["y"]),
                (G.nodes[v]["x"], G.nodes[v]["y"]),
            ])
        edges.append(nx_d)

    edges_gdf = gpd.GeoDataFrame(edges, geometry="geometry")
    edges_gdf.index = pd.MultiIndex.from_tuples(
        [(r["u"], r["v"], r["key"]) for _, r in edges_gdf.iterrows()],
        names=["u", "v", "key"],
    )
    return nodes_gdf, edges_gdf


def _mock_nearest_nodes(G, lon, lat):
    # Return the node whose (x,y) is closest
    best, best_d = None, float("inf")
    for n, d in G.nodes(data=True):
        dist = (d["x"] - lon) ** 2 + (d["y"] - lat) ** 2
        if dist < best_d:
            best, best_d = n, dist
    return best


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetColor:
    def test_free_flow(self):
        assert _get_color(0.0) == "#34a853"

    def test_heavy(self):
        color = _get_color(0.8)
        assert color.startswith("#")
        assert len(color) == 7

    def test_clamps_above_1(self):
        c1 = _get_color(1.0)
        c2 = _get_color(2.0)
        assert c1 == c2

    def test_clamps_below_0(self):
        c1 = _get_color(0.0)
        c2 = _get_color(-1.0)
        assert c1 == c2


class TestRoutingServiceEdgeCosts:
    """Test internal edge weight computation without full routing."""

    def setup_method(self):
        import unittest.mock as mock
        import osmnx as ox
        self._ox_patch = mock.patch.object(ox, "graph_to_gdfs", side_effect=_mock_graph_to_gdfs)
        self._ox_patch.start()

        G = _make_small_graph()
        node_list = list(G.nodes)
        cfg = _make_cfg()
        self.svc = RoutingService(G, node_list, cfg)

    def teardown_method(self):
        self._ox_patch.stop()

    def test_weights_are_positive(self):
        traffic = np.full(5, 0.3)   # 30% traffic
        weights = self.svc._compute_edge_weights(traffic, cost_type="time")
        for key, val in weights.items():
            assert val > 0, f"Edge {key} has non-positive cost"

    def test_high_traffic_increases_cost(self):
        """Higher traffic should mean higher travel time cost."""
        low_t  = np.full(5, 0.1)
        high_t = np.full(5, 0.9)
        w_low  = self.svc._compute_edge_weights(low_t,  cost_type="time")
        w_high = self.svc._compute_edge_weights(high_t, cost_type="time")

        # At least some edges should have higher cost at high traffic
        costs_increased = sum(
            1 for k in w_low
            if w_high[k] > w_low[k]
        )
        assert costs_increased > 0

    def test_balanced_cost_different_from_time(self):
        traffic = np.full(5, 0.5)
        w_time  = self.svc._compute_edge_weights(traffic, cost_type="time")
        w_bal   = self.svc._compute_edge_weights(traffic, cost_type="balanced")
        # At least some edge costs should differ
        differ = any(
            abs(w_bal[k] - w_time[k]) > 1e-6
            for k in w_time
        )
        assert differ


class TestRoutingServiceGetRoutes:
    def setup_method(self):
        import unittest.mock as mock
        import osmnx as ox
        self._gdfs_patch = mock.patch.object(
            ox, "graph_to_gdfs", side_effect=_mock_graph_to_gdfs
        )
        self._nn_patch = mock.patch.object(
            ox.distance, "nearest_nodes", side_effect=_mock_nearest_nodes
        )
        self._gdfs_patch.start()
        self._nn_patch.start()

        self.G = _make_small_graph()
        node_list = list(self.G.nodes)
        cfg = _make_cfg()
        self.svc = RoutingService(self.G, node_list, cfg)

    def teardown_method(self):
        self._gdfs_patch.stop()
        self._nn_patch.stop()

    def test_returns_at_least_one_route(self):
        cur  = np.full(5, 0.3)
        pred = np.full(5, 0.5)
        routes = self.svc.get_routes("NodeA", "NodeD", cur, pred)
        assert len(routes) >= 1

    def test_route_has_required_fields(self):
        cur   = np.full(5, 0.3)
        pred  = np.full(5, 0.5)
        routes = self.svc.get_routes("NodeA", "NodeD", cur, pred)
        for r in routes:
            assert isinstance(r, RouteResult)
            assert r.distance_km > 0
            assert r.eta_now_min > 0
            assert r.eta_future_min > 0
            assert 0.0 <= r.congestion_score <= 1.0
            assert r.label in ("Fastest Now", "Fastest 15-min", "Balanced")

    def test_high_traffic_increases_eta_future(self):
        """Predicted ETA should increase with higher predicted traffic."""
        low  = np.full(5, 0.1)
        high = np.full(5, 0.9)

        routes_low  = self.svc.get_routes("NodeA", "NodeD", low, low)
        routes_high = self.svc.get_routes("NodeA", "NodeD", low, high)

        # Find "Fastest 15-min" in both
        def _get(routes, label):
            for r in routes:
                if r.label == label:
                    return r
            return None

        r_low  = _get(routes_low,  "Fastest 15-min")
        r_high = _get(routes_high, "Fastest 15-min")

        if r_low and r_high:
            assert r_high.eta_future_min >= r_low.eta_future_min

    def test_same_origin_dest_raises(self):
        with pytest.raises(ValueError):
            self.svc.get_routes("NodeA", "NodeA", np.zeros(5), np.zeros(5))

    def test_unknown_origin_raises(self):
        with pytest.raises(ValueError):
            self.svc.get_routes("Unknown", "NodeD", np.zeros(5), np.zeros(5))

    def test_location_names(self):
        names = self.svc.location_names()
        assert "NodeA" in names
        assert "NodeD" in names
