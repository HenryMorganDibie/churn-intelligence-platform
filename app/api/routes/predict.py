"""
Prediction Routes — Real-Time Single Customer Scoring
"""

from fastapi import APIRouter, HTTPException
from app.core.schemas import CustomerEvent, ChurnPrediction
from app.features.engineer import engineer_features
from app.models.predictor import predict_single
from app.core.logger import logger

router = APIRouter()


@router.post("/", response_model=ChurnPrediction)
async def score_customer(event: CustomerEvent):
    """
    Score a single customer for churn risk.
    Returns probability, risk segment, SHAP-driven top risk factors, and recommended action.
    """
    try:
        features_df = engineer_features(event.model_dump())
        prediction = predict_single(event.customer_id, features_df)
        logger.info(
            f"Scored {event.customer_id} → {prediction.risk_segment} "
            f"({prediction.churn_probability:.2%})"
        )
        return prediction
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run python -m app.models.trainer first.",
        )
    except Exception as e:
        logger.error(f"Prediction failed for {event.customer_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
