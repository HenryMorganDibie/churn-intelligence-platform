"""
Health Check Routes

/health/        liveness: the process is up (Kubernetes livenessProbe)
/health/ready   readiness: the model is loaded and can score (Kubernetes readinessProbe)
/health/model   model metadata, metrics and provenance (file or MLflow registry alias)
"""

from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/")
async def health_check():
    return {
        "status": "healthy",
        "service": "Churn Intelligence Platform",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/ready")
async def readiness():
    try:
        from app.models.predictor import load_artifacts
        load_artifacts()
        return {"ready": True}
    except Exception as e:
        return JSONResponse(status_code=503, content={"ready": False, "error": str(e)})


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
            "provenance": artifacts.get("provenance"),
        }
    except Exception as e:
        return {"model_loaded": False, "error": str(e)}
