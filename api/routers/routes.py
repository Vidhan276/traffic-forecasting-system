"""POST /routes — future-aware multi-route A→B routing."""

import logging
import time
from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, HTTPException, Request

from api.schemas import (
    RouteRequest,
    RouteResultSchema,
    RoutesResponse,
    SegmentInfoSchema,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Routing"])
FORECAST_CACHE_TTL_SEC = 300.0


@router.post(
    "/routes",
    response_model=RoutesResponse,
    summary="Get multiple route options with traffic forecasts",
)
async def get_routes(
    body: RouteRequest,
    request: Request,
) -> RoutesResponse:
    """
    Return up to 3 route alternatives between origin and destination:

    1. **Fastest Now** — Dijkstra minimising *current* travel time.
    2. **Fastest 15-min** — Dijkstra minimising *predicted* future travel time.
    3. **Balanced** — minimises a blend of distance and predicted congestion.

    Each route includes per-segment colour coding and both current/future ETAs.
    """
    t_start = time.perf_counter()

    forecast_svc = request.app.state.forecast_svc
    routing_svc  = request.app.state.routing_svc

    # Validate location names
    available = routing_svc.location_names()
    if body.origin not in available:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown origin '{body.origin}'. Available: {available}",
        )
    if body.destination not in available:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown destination '{body.destination}'. Available: {available}",
        )

    try:
        # ── 1. Run GNN forecast ──────────────────────────────────────────────
        cached_forecast = getattr(request.app.state, "forecast_cache", None)
        now_mono = time.monotonic()
        if cached_forecast and now_mono - cached_forecast["created_at"] < FORECAST_CACHE_TTL_SEC:
            pred_unnorm = cached_forecast["pred_unnorm"]
            cur_traffic = cached_forecast["cur_traffic"]
        else:
            pred_unnorm = forecast_svc.predict()              # (num_nodes, pred_len)
            cur_traffic = forecast_svc.get_current_traffic()  # (num_nodes,)
            request.app.state.forecast_cache = {
                "created_at": now_mono,
                "pred_unnorm": pred_unnorm,
                "cur_traffic": cur_traffic,
            }

        horizon_idx = min(
            max(int(round(body.horizon_minutes / 5)) - 1, 0),
            pred_unnorm.shape[1] - 1,
        )

        # Keep current and future congestion on the same absolute scale. Separate
        # min/max normalization makes both arrays span 0..1 independently, which
        # hides whether traffic is actually getting better or worse.
        max_observed = max(
            float(np.nanmax(cur_traffic)),
            float(np.nanmax(pred_unnorm[:, : horizon_idx + 1])),
            1e-6,
        )
        cur_norm = np.clip(cur_traffic / max_observed, 0.0, 1.0)
        pred_norm = np.clip(pred_unnorm[:, horizon_idx] / max_observed, 0.0, 1.0)

        # ── 2. Compute routes ────────────────────────────────────────────────
        route_results = routing_svc.get_routes(
            origin_name=body.origin,
            dest_name=body.destination,
            current_traffic=cur_norm,
            predicted_traffic=pred_norm,
        )

        if not route_results:
            raise HTTPException(
                status_code=404,
                detail=f"No route found between '{body.origin}' and '{body.destination}'.",
            )

        # ── 3. Serialise to Pydantic schema ──────────────────────────────────
        route_schemas = []
        for r in route_results:
            segments = [
                SegmentInfoSchema(
                    coords=seg.coords,
                    color=seg.color,
                    congestion=seg.congestion,
                    road_name=seg.road_name,
                    highway=seg.highway,
                )
                for seg in r.segments
            ]
            route_schemas.append(
                RouteResultSchema(
                    label=r.label,
                    color=r.color,
                    distance_km=r.distance_km,
                    eta_now_min=r.eta_now_min,
                    eta_future_min=r.eta_future_min,
                    congestion_score=r.congestion_score,
                    segments=segments,
                )
            )

        latency_ms = (time.perf_counter() - t_start) * 1000

        logger.info(
            "Route request: %s -> %s | routes=%d | latency=%.0f ms",
            body.origin,
            body.destination,
            len(route_schemas),
            latency_ms,
        )

        return RoutesResponse(
            origin=body.origin,
            destination=body.destination,
            routes=route_schemas,
            computed_at=datetime.now(timezone.utc).isoformat(),
            latency_ms=round(latency_ms, 1),
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Error in /routes: %s → %s", body.origin, body.destination)
        raise HTTPException(status_code=500, detail=str(e))
