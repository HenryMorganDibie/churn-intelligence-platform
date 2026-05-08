"""
Batch Scoring Routes — Score multiple customers in one request
"""

import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from app.core.schemas import BatchRequest, BatchResponse
from app.features.engineer import engineer_batch_features
from app.models.predictor import predict_batch
from app.core.logger import logger

router = APIRouter()


@router.post("/score", response_model=BatchResponse)
async def batch_score(request: BatchRequest):
    """
    Score a batch of customers. Supports up to 10,000 records per request.
    SHAP explanations are omitted for batch performance; use /predict for individual explanations.
    """
    if len(request.customers) > 10_000:
        raise HTTPException(status_code=400, detail="Max 10,000 customers per batch request.")

    try:
        records = [c.model_dump() for c in request.customers]
        customer_ids = [c["customer_id"] for c in records]

        features_df = engineer_batch_features(records)
        predictions = predict_batch(customer_ids, features_df)

        high_risk = sum(1 for p in predictions if p.risk_segment in ("high", "critical"))

        logger.info(
            f"Batch scored {len(predictions)} customers. "
            f"High/Critical risk: {high_risk}"
        )

        return BatchResponse(
            total_scored=len(predictions),
            high_risk_count=high_risk,
            predictions=predictions,
            job_id=str(uuid.uuid4()),
            completed_at=datetime.now(timezone.utc),
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run python -m app.models.trainer first.",
        )
    except Exception as e:
        logger.error(f"Batch scoring failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
