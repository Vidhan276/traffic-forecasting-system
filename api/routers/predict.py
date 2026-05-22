"""POST /predict_nodes — run the GNN and return per-node traffic forecasts."""

import logging
from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, HTTPException, Request

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
