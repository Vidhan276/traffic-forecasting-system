"""
schemas.py — Pydantic request/response models for the FastAPI endpoints.

All models use Python type hints and Pydantic v2 validation.
They serve as both API documentation (via OpenAPI) and runtime input validation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, model_validator


# ═══════════════════════════════════════════════════════════════════════════════
# /health
# ═══════════════════════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    """Response from GET /health."""
    status:      str   = Field("ok", description="Always 'ok' if server is healthy.")
    model_type:  str   = Field(..., description="Loaded model architecture name.")
    num_nodes:   int   = Field(..., description="Number of nodes in the loaded graph.")
    device:      str   = Field(..., description="Compute device: 'cpu' or 'cuda'.")
    uptime_info: str   = Field(..., description="Server startup timestamp.")


# ═══════════════════════════════════════════════════════════════════════════════
# /predict_nodes
# ═══════════════════════════════════════════════════════════════════════════════

class PredictNodesRequest(BaseModel):
    """
    Request body for POST /predict_nodes.

    If traffic_seq is omitted, the service loads the latest available sequence
    from disk automatically (recommended for production use).
    """
    traffic_seq: Optional[list[list[float]]] = Field(
        default=None,
        description=(
            "Past traffic values, shape (seq_len, num_nodes). "
            "If omitted, the server reads the latest data from disk."
        ),
    )
    horizon_steps: int = Field(
        default=3,
        ge=1,
        le=12,
        description="Number of future 5-minute steps to forecast (max 12).",
    )


class PredictNodesResponse(BaseModel):
    """Response from POST /predict_nodes."""
    num_nodes:     int
    horizon_steps: int
    # Outer list: pred_len timesteps. Inner list: one value per node.
    forecasts: list[list[float]] = Field(
        description="Shape: (horizon_steps, num_nodes). Each inner list is one future timestep."
    )
    computed_at: str


# ═══════════════════════════════════════════════════════════════════════════════
# /routes
# ═══════════════════════════════════════════════════════════════════════════════

class RouteRequest(BaseModel):
    """Request body for POST /routes."""
    origin:           str = Field(..., description="Origin location name (e.g., 'Kothrud').")
    destination:      str = Field(..., description="Destination location name (e.g., 'Hinjewadi').")
    horizon_minutes:  int = Field(default=15, ge=5, le=60, description="Forecast horizon in minutes.")

    @model_validator(mode="after")
    def origin_dest_differ(self) -> "RouteRequest":
        if self.origin == self.destination:
            raise ValueError("origin and destination must be different locations.")
        return self


class SegmentInfoSchema(BaseModel):
    """Per-road-segment display info for the frontend map."""
    coords:     list[tuple[float, float]]
    color:      str   = Field(..., description="Hex colour string based on congestion level.")
    congestion: float = Field(..., ge=0.0, le=1.0)
    road_name:  str
    highway:    str


class RouteResultSchema(BaseModel):
    """A single route option."""
    label:            str   = Field(..., description="Route variant name.")
    color:            str   = Field(..., description="Route line colour on the overview map.")
    distance_km:      float
    eta_now_min:      float = Field(..., description="Travel time with current traffic (minutes).")
    eta_future_min:   float = Field(..., description="Travel time with predicted traffic (minutes).")
    congestion_score: float = Field(..., ge=0.0, le=1.0, description="Mean predicted congestion [0-1].")
    segments:         list[SegmentInfoSchema]


class RoutesResponse(BaseModel):
    """Response from POST /routes."""
    origin:      str
    destination: str
    routes:      list[RouteResultSchema]
    computed_at: str
    latency_ms:  float = Field(..., description="Server-side computation time in milliseconds.")
