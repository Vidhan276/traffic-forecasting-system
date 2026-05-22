"""
api/main.py — FastAPI application entry point.

Startup sequence:
  1. Load config.yaml
  2. Load graph + model via ForecastingService.load()
  3. Initialise RoutingService (caches graph GDFs)
  4. Mount routers: /health, /predict_nodes, /routes

The model and graph are loaded ONCE and stored on app.state.
All inference paths use torch.no_grad() (enforced inside ForecastingService).

Run locally:
    py -3.13 -m uvicorn api.main:app --reload --port 8000

Run in production:
    py -3.13 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── Ensure project root is importable ─────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from api.routers import health, predict, routes
from data.loader import load_config, load_graph
from services.forecasting import ForecastingService
from services.routing import RoutingService

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("api.main")


def _warm_route_cache_background(routing_svc, cur_norm, pred_norm) -> None:
    locations = routing_svc.location_names()
    warmed = 0
    for origin in locations:
        for destination in locations:
            if origin == destination:
                continue
            try:
                routing_svc.get_routes(origin, destination, cur_norm, pred_norm)
                warmed += 1
            except Exception as exc:
                logger.debug("Route cache warmup skipped %s -> %s: %s", origin, destination, exc)
    logger.info("Background route cache warmup complete: %d pairs", warmed)
    import gc
    gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# Lifespan — load everything once at startup
# ═══════════════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and services on startup; clean up on shutdown."""
    import torch
    import gc
    torch.set_num_threads(1)
    gc.collect()

    logger.info("=" * 60)
    logger.info("Traffic Forecasting API — starting up")
    logger.info("=" * 60)

    cfg = load_config()

    # ── Forecasting service ──────────────────────────────────────────────────
    forecast_svc = ForecastingService(cfg)
    try:
        forecast_svc.load()
    except FileNotFoundError as e:
        logger.error(
            "Model or graph not found: %s\n"
            "Run `py -3.13 ml/train_model.py` first.",
            e,
        )
        # Still start the server so /health reports a meaningful error state
        app.state.forecast_svc = None
        app.state.routing_svc  = None
        app.state.started_at   = datetime.now(timezone.utc).isoformat()
        yield
        return

    app.state.forecast_svc = forecast_svc

    # ── Routing service ──────────────────────────────────────────────────────
    logger.info("Initialising RoutingService...")
    G_route, _ = load_graph(cfg, key="full_pune_graph")
    routing_svc  = RoutingService(G_route, forecast_svc.node_list, cfg)
    app.state.routing_svc = routing_svc

    logger.info("Warming forecast and default route cache...")
    pred_unnorm = forecast_svc.predict()
    cur_traffic = forecast_svc.get_current_traffic()
    max_observed = max(
        float(np.nanmax(cur_traffic)),
        float(np.nanmax(pred_unnorm)),
        1e-6,
    )
    cur_norm = np.clip(cur_traffic / max_observed, 0.0, 1.0)
    pred_norm = np.clip(pred_unnorm[:, min(2, pred_unnorm.shape[1] - 1)] / max_observed, 0.0, 1.0)
    app.state.forecast_cache = {
        "created_at": time.monotonic(),
        "pred_unnorm": pred_unnorm,
        "cur_traffic": cur_traffic,
    }
    locations = routing_svc.location_names()
    if "Kothrud" in locations and "Hinjewadi" in locations:
        routing_svc.get_routes("Kothrud", "Hinjewadi", cur_norm, pred_norm)
    threading.Thread(
        target=_warm_route_cache_background,
        args=(routing_svc, cur_norm, pred_norm),
        daemon=True,
        name="route-cache-warmup",
    ).start()

    app.state.started_at = datetime.now(timezone.utc).isoformat()

    logger.info("=" * 60)
    logger.info("API ready.  Endpoints:")
    logger.info("  GET  /health")
    logger.info("  POST /predict_nodes")
    logger.info("  POST /routes")
    logger.info("  GET  /docs  (interactive OpenAPI docs)")
    logger.info("=" * 60)

    yield  # ← server runs here

    logger.info("Traffic Forecasting API — shutting down")


# ═══════════════════════════════════════════════════════════════════════════════
# App factory
# ═══════════════════════════════════════════════════════════════════════════════

def create_app() -> FastAPI:
    app = FastAPI(
        title="Traffic Forecasting API",
        description=(
            "GNN-based traffic forecasting and future-aware routing for Pune / Kothrud. "
            "Provides per-node traffic predictions and multi-route A→B planning."
        ),
        version="2.0.0",
        lifespan=lifespan,
    )

    # Allow Streamlit (on port 8501) to call the API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount routers
    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(routes.router)

    return app


# Single app instance used by uvicorn
app = create_app()
