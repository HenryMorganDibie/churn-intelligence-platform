"""
Health Check Routes
"""

from fastapi import APIRouter
from datetime import datetime, timezone

router = APIRouter()


@router.get("/")
async def health_check():
    return {
        "status": "healthy",
        "service": "Churn Intelligence Platform",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/model")
async def model_status():
    try:
        from app.models.predictor import load_artifacts
        artifacts = load_artifacts()
        metrics = artifacts.get("metrics", {})
        return {
            "model_loaded": True,
            "auc_roc": metrics.get("auc"),
            "brier_score": metrics.get("brier"),
        }
    except Exception as e:
        return {"model_loaded": False, "error": str(e)}
