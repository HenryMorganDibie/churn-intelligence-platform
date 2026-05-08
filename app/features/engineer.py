"""
Feature Engineering — Time-Aware Feature Store
Computes derived features from raw customer event data.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from app.core.logger import logger


FEATURE_COLUMNS = [
    "tenure_days",
    "monthly_charges",
    "total_charges",
    "num_support_tickets",
    "num_logins_last_30d",
    "avg_session_duration_min",
    "plan_changes_last_90d",
    "payment_failures_last_6m",
    "has_contract",
    "is_autopay",
    # Engineered
    "charge_per_day",
    "support_intensity",
    "engagement_score",
    "payment_risk_score",
    "loyalty_score",
    "product_tier_encoded",
]

TIER_MAP = {"basic": 0, "standard": 1, "premium": 2}


def engineer_features(raw: Dict[str, Any]) -> pd.DataFrame:
    """
    Transform raw customer event dict into model-ready feature DataFrame.
    Returns a single-row DataFrame aligned to FEATURE_COLUMNS.
    """
    tenure = max(raw["tenure_days"], 1)  # prevent div by zero

    charge_per_day = raw["total_charges"] / tenure
    support_intensity = raw["num_support_tickets"] / max(tenure / 30, 1)

    # Engagement: normalised login frequency × session depth
    login_norm = min(raw["num_logins_last_30d"] / 30, 1.0)
    session_norm = min(raw["avg_session_duration_min"] / 60, 1.0)
    engagement_score = round((login_norm * 0.6) + (session_norm * 0.4), 4)

    # Payment risk: failures weighted by recency proxy (plan changes signal instability)
    payment_risk_score = round(
        (raw["payment_failures_last_6m"] * 0.7) + (raw["plan_changes_last_90d"] * 0.3), 4
    )

    # Loyalty: tenure + contract + autopay signals
    loyalty_score = round(
        (min(tenure / 365, 1.0) * 0.5)
        + (0.3 if raw["has_contract"] else 0)
        + (0.2 if raw["is_autopay"] else 0),
        4,
    )

    product_tier_encoded = TIER_MAP.get(raw.get("product_tier", "basic"), 0)

    features = {
        "tenure_days": tenure,
        "monthly_charges": raw["monthly_charges"],
        "total_charges": raw["total_charges"],
        "num_support_tickets": raw["num_support_tickets"],
        "num_logins_last_30d": raw["num_logins_last_30d"],
        "avg_session_duration_min": raw["avg_session_duration_min"],
        "plan_changes_last_90d": raw["plan_changes_last_90d"],
        "payment_failures_last_6m": raw["payment_failures_last_6m"],
        "has_contract": int(raw["has_contract"]),
        "is_autopay": int(raw["is_autopay"]),
        "charge_per_day": round(charge_per_day, 4),
        "support_intensity": round(support_intensity, 4),
        "engagement_score": engagement_score,
        "payment_risk_score": payment_risk_score,
        "loyalty_score": loyalty_score,
        "product_tier_encoded": product_tier_encoded,
    }

    df = pd.DataFrame([features])[FEATURE_COLUMNS]
    logger.debug(f"Engineered features for customer: {raw.get('customer_id')}")
    return df


def engineer_batch_features(records: list[Dict[str, Any]]) -> pd.DataFrame:
    """Engineer features for a list of customer records."""
    frames = [engineer_features(r) for r in records]
    return pd.concat(frames, ignore_index=True)
