"""GET /health — simple liveness check."""

from datetime import datetime, timezone

from fastapi import APIRouter, Request

from api.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health(request: Request) -> HealthResponse:
    """
    Returns server health status, loaded model type, graph size, and device.
    Useful for monitoring and load-balancer liveness probes.
    """
    svc = request.app.state.forecast_svc
    return HealthResponse(
        status="ok",
        model_type=svc.cfg["model"]["type"],
        num_nodes=len(svc.node_list) if svc.node_list else 0,
        device=str(svc._device),
        uptime_info=request.app.state.started_at,
    )
