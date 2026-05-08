"""
Model Trainer — XGBoost + SHAP
Run this script to train the churn model on synthetic or real data.
Output: models/churn_model.pkl
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    roc_auc_score, classification_report, brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import shap
from app.features.engineer import FEATURE_COLUMNS
from app.core.logger import logger


def generate_synthetic_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic churn data for development and testing.
    In production, replace with your actual data ingestion pipeline.
    """
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "tenure_days": rng.integers(7, 1095, n),
        "monthly_charges": rng.uniform(20, 150, n).round(2),
        "total_charges": rng.uniform(50, 5000, n).round(2),
        "num_support_tickets": rng.integers(0, 15, n),
        "num_logins_last_30d": rng.integers(0, 60, n),
        "avg_session_duration_min": rng.uniform(0, 90, n).round(2),
        "plan_changes_last_90d": rng.integers(0, 5, n),
        "payment_failures_last_6m": rng.integers(0, 6, n),
        "has_contract": rng.integers(0, 2, n),
        "is_autopay": rng.integers(0, 2, n),
        "product_tier": rng.choice(["basic", "standard", "premium"], n),
    })

    # Engineered columns (simplified for training data gen)
    df["charge_per_day"] = (df["total_charges"] / df["tenure_days"]).round(4)
    df["support_intensity"] = (df["num_support_tickets"] / (df["tenure_days"] / 30).clip(1)).round(4)
    login_norm = (df["num_logins_last_30d"] / 30).clip(0, 1)
    session_norm = (df["avg_session_duration_min"] / 60).clip(0, 1)
    df["engagement_score"] = (login_norm * 0.6 + session_norm * 0.4).round(4)
    df["payment_risk_score"] = (df["payment_failures_last_6m"] * 0.7 + df["plan_changes_last_90d"] * 0.3).round(4)
    df["loyalty_score"] = (
        (df["tenure_days"] / 365).clip(0, 1) * 0.5
        + df["has_contract"] * 0.3
        + df["is_autopay"] * 0.2
    ).round(4)
    tier_map = {"basic": 0, "standard": 1, "premium": 2}
    df["product_tier_encoded"] = df["product_tier"].map(tier_map)

    # Churn label: probabilistic rule based on features
    churn_score = (
        df["payment_risk_score"] * 0.35
        + df["support_intensity"] * 0.25
        - df["loyalty_score"] * 0.25
        - df["engagement_score"] * 0.15
        + rng.normal(0, 0.1, n)
    )
    df["churn"] = (churn_score > churn_score.median()).astype(int)

    return df[FEATURE_COLUMNS + ["churn"]]


def train():
    logger.info("Loading training data...")
    df = generate_synthetic_data(n=8000)

    X = df[FEATURE_COLUMNS]
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    logger.info("Training XGBoost model...")
    base_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)

    # Fit base model first (needed for SHAP)
    base_model.fit(X_train, y_train)

    # Calibrate on held-out test set using prefit mode
    model = CalibratedClassifierCV(FrozenEstimator(base_model), method="isotonic")
    model.fit(X_test, y_test)

    # Evaluate
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)

    logger.info(f"AUC-ROC: {auc:.4f} | Brier Score: {brier:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred))

    # SHAP explainer on base estimator
    logger.info("Building SHAP explainer...")
    explainer = shap.TreeExplainer(base_model)

    # Save artifacts
    os.makedirs("models", exist_ok=True)
    artifacts = {
        "model": model,
        "explainer": explainer,
        "feature_columns": FEATURE_COLUMNS,
        "training_stats": X_train.describe().to_dict(),
        "metrics": {"auc": auc, "brier": brier},
    }
    with open("models/churn_model.pkl", "wb") as f:
        pickle.dump(artifacts, f)

    logger.info("Model saved to models/churn_model.pkl")
    return artifacts


if __name__ == "__main__":
    train()
