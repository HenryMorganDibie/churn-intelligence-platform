"""
Model Predictor — Loads trained artifacts and scores customers
"""

import pickle
import numpy as np
import pandas as pd
import shap
from functools import lru_cache
from datetime import datetime, timezone
from typing import Dict, Any

from app.core.config import settings
from app.core.schemas import ChurnPrediction
from app.core.logger import logger


@lru_cache(maxsize=1)
def load_artifacts() -> Dict[str, Any]:
    """Load model artifacts once and cache in memory."""
    try:
        with open(settings.MODEL_PATH, "rb") as f:
            artifacts = pickle.load(f)
        logger.info("Model artifacts loaded successfully.")
        return artifacts
    except FileNotFoundError:
        logger.error(f"Model not found at {settings.MODEL_PATH}. Run trainer first.")
        raise


def get_risk_segment(prob: float) -> str:
    if prob < 0.30:
        return "low"
    elif prob < 0.55:
        return "medium"
    elif prob < 0.75:
        return "high"
    return "critical"


def get_recommended_action(segment: str) -> str:
    actions = {
        "low": "No action required. Monitor monthly.",
        "medium": "Send engagement nudge. Offer loyalty incentive.",
        "high": "Trigger retention workflow. Assign CSM outreach.",
        "critical": "Immediate intervention required. Escalate to retention team.",
    }
    return actions[segment]


def predict_single(customer_id: str, features_df: pd.DataFrame) -> ChurnPrediction:
    """Score a single customer and return a ChurnPrediction."""
    artifacts = load_artifacts()
    model = artifacts["model"]
    explainer = artifacts["explainer"]
    feature_cols = artifacts["feature_columns"]

    X = features_df[feature_cols]
    prob = float(model.predict_proba(X)[0, 1])
    segment = get_risk_segment(prob)

    # SHAP values for top drivers
    try:
        base_estimator = model.estimators_[0] if hasattr(model, "estimators_") else model
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        shap_series = pd.Series(shap_values[0], index=feature_cols)
        top_drivers = (
            shap_series.abs()
            .nlargest(5)
            .index.tolist()
        )
        top_risk_drivers = {feat: round(float(shap_series[feat]), 4) for feat in top_drivers}
    except Exception as e:
        logger.warning(f"SHAP failed for {customer_id}: {e}")
        top_risk_drivers = {}

    return ChurnPrediction(
        customer_id=customer_id,
        churn_probability=round(prob, 4),
        risk_segment=segment,
        top_risk_drivers=top_risk_drivers,
        recommended_action=get_recommended_action(segment),
        scored_at=datetime.now(timezone.utc),
    )


def predict_batch(customer_ids: list[str], features_df: pd.DataFrame) -> list[ChurnPrediction]:
    """Score a batch of customers."""
    artifacts = load_artifacts()
    model = artifacts["model"]
    feature_cols = artifacts["feature_columns"]
    X = features_df[feature_cols]

    probs = model.predict_proba(X)[:, 1]
    predictions = []

    for i, (cid, prob) in enumerate(zip(customer_ids, probs)):
        segment = get_risk_segment(float(prob))
        predictions.append(
            ChurnPrediction(
                customer_id=cid,
                churn_probability=round(float(prob), 4),
                risk_segment=segment,
                top_risk_drivers={},  # omit SHAP in batch for speed; can enable selectively
                recommended_action=get_recommended_action(segment),
                scored_at=datetime.now(timezone.utc),
            )
        )

    return predictions
