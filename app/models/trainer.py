"""
Model Trainer: XGBoost + isotonic calibration + SHAP, tracked in MLflow.

Run:  python -m app.models.trainer
Output:
  models/churn_model.pkl            local artifact bundle (used by the API by default)
  MLflow run                        params, metrics, bundle, sklearn model, feature list
  MLflow registry "churn-xgb"       new version tagged as challenger; promoted to the
                                    "champion" alias only if it passes the promotion gate

Data split (no leakage):
  train 60%  -> fit XGBoost
  calib 20%  -> fit isotonic calibration on the frozen booster
  test  20%  -> report AUC / Brier; never seen by either fit
"""

from __future__ import annotations

import os
import pickle
import tempfile
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from app.core.config import settings
from app.core.logger import logger
from app.features.engineer import FEATURE_COLUMNS

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}


def generate_synthetic_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic churn data for development and testing.
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

    churn_score = (
        df["payment_risk_score"] * 0.35
        + df["support_intensity"] * 0.25
        - df["loyalty_score"] * 0.25
        - df["engagement_score"] * 0.15
        + rng.normal(0, 0.1, n)
    )
    df["churn"] = (churn_score > churn_score.median()).astype(int)

    return df[FEATURE_COLUMNS + ["churn"]]


@dataclass
class GateDecision:
    promoted: bool
    reason: str


def promotion_gate(candidate: dict, champion: dict | None,
                   min_auc_gain: float, max_brier_increase: float) -> GateDecision:
    """
    Champion/challenger rule. A candidate replaces the champion only if its held-out
    AUC is at least `min_auc_gain` higher AND its Brier score is not worse by more than
    `max_brier_increase`. With no champion yet, the first model is promoted.
    """
    if champion is None:
        return GateDecision(True, "no existing champion")
    auc_gain = candidate["test_auc"] - champion["test_auc"]
    brier_increase = candidate["test_brier"] - champion["test_brier"]
    if auc_gain < min_auc_gain:
        return GateDecision(False, f"AUC gain {auc_gain:+.4f} below required {min_auc_gain:+.4f}")
    if brier_increase > max_brier_increase:
        return GateDecision(False, f"Brier worsened by {brier_increase:+.4f} (limit {max_brier_increase:+.4f})")
    return GateDecision(True, f"AUC gain {auc_gain:+.4f}, Brier change {brier_increase:+.4f}")


def fit_and_evaluate(df: pd.DataFrame, params: dict | None = None, seed: int = 42) -> dict:
    """Fit booster on train, calibrate on calib, evaluate on test. Returns the artifact bundle."""
    params = {**XGB_PARAMS, **(params or {})}
    X, y = df[FEATURE_COLUMNS], df["churn"]

    X_train, X_hold, y_train, y_hold = train_test_split(X, y, test_size=0.4, stratify=y, random_state=seed)
    X_calib, X_test, y_calib, y_test = train_test_split(X_hold, y_hold, test_size=0.5, stratify=y_hold, random_state=seed)

    booster = xgb.XGBClassifier(**params)
    booster.fit(X_train, y_train)

    model = CalibratedClassifierCV(FrozenEstimator(booster), method="isotonic")
    model.fit(X_calib, y_calib)

    y_prob = model.predict_proba(X_test)[:, 1]
    raw_prob = booster.predict_proba(X_test)[:, 1]
    metrics = {
        "test_auc": float(roc_auc_score(y_test, y_prob)),
        "test_brier": float(brier_score_loss(y_test, y_prob)),
        "test_brier_uncalibrated": float(brier_score_loss(y_test, raw_prob)),
        "base_rate": float(y.mean()),
        "n_train": int(len(X_train)),
        "n_calib": int(len(X_calib)),
        "n_test": int(len(X_test)),
    }
    logger.info("\n" + classification_report(y_test, (y_prob >= 0.5).astype(int)))

    return {
        "model": model,
        "explainer": shap.TreeExplainer(booster),
        "feature_columns": FEATURE_COLUMNS,
        "training_stats": X_train.describe().to_dict(),
        "reference_sample": X_train.sample(min(2000, len(X_train)), random_state=seed),
        "metrics": {**metrics, "auc": metrics["test_auc"], "brier": metrics["test_brier"]},
        "params": params,
    }


def _save_bundle(artifacts: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(artifacts, f)


def log_to_mlflow(artifacts: dict, data_meta: dict) -> dict:
    """Log the run, register a new version, and apply the champion/challenger gate."""
    import mlflow
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT)
    client = MlflowClient()
    name = settings.MLFLOW_MODEL_NAME

    with mlflow.start_run() as run:
        mlflow.log_params({f"xgb_{k}": v for k, v in artifacts["params"].items()})
        mlflow.log_params(data_meta)
        mlflow.log_metrics({k: v for k, v in artifacts["metrics"].items() if k not in ("auc", "brier")})
        mlflow.set_tags({"model_family": "xgboost+isotonic", "explainer": "shap.TreeExplainer"})

        with tempfile.TemporaryDirectory() as tmp:
            bundle_path = os.path.join(tmp, "churn_model.pkl")
            _save_bundle(artifacts, bundle_path)
            mlflow.log_artifact(bundle_path, artifact_path="bundle")

        sample = artifacts["reference_sample"].head(50)
        signature = infer_signature(sample, artifacts["model"].predict_proba(sample)[:, 1])
        mlflow.sklearn.log_model(artifacts["model"], artifact_path="model", signature=signature,
                                 input_example=sample.head(3))

        version = mlflow.register_model(f"runs:/{run.info.run_id}/model", name).version

    champion = None
    try:
        champ_mv = client.get_model_version_by_alias(name, "champion")
        champ_run = client.get_run(champ_mv.run_id)
        champion = {"test_auc": champ_run.data.metrics["test_auc"],
                    "test_brier": champ_run.data.metrics["test_brier"],
                    "version": champ_mv.version}
    except Exception:
        champion = None

    decision = promotion_gate(artifacts["metrics"], champion,
                              settings.PROMOTION_MIN_AUC_GAIN, settings.PROMOTION_MAX_BRIER_INCREASE)
    client.set_model_version_tag(name, version, "gate_decision", "promoted" if decision.promoted else "rejected")
    client.set_model_version_tag(name, version, "gate_reason", decision.reason)
    client.set_registered_model_alias(name, "challenger", version)
    if decision.promoted:
        client.set_registered_model_alias(name, "champion", version)

    logger.info(f"Registered {name} v{version}: {'PROMOTED to champion' if decision.promoted else 'kept as challenger'} ({decision.reason})")
    return {"run_id": run.info.run_id, "version": version, "promoted": decision.promoted,
            "reason": decision.reason, "previous_champion": champion}


def train(n_rows: int = 8000, seed: int = 42, params: dict | None = None, use_mlflow: bool | None = None) -> dict:
    logger.info("Loading training data...")
    df = generate_synthetic_data(n=n_rows, seed=seed)
    artifacts = fit_and_evaluate(df, params=params, seed=seed)
    m = artifacts["metrics"]
    logger.info(f"Held-out AUC {m['test_auc']:.4f} | Brier {m['test_brier']:.4f} "
                f"(uncalibrated {m['test_brier_uncalibrated']:.4f})")

    _save_bundle(artifacts, settings.MODEL_PATH)
    logger.info(f"Model saved to {settings.MODEL_PATH}")

    if use_mlflow if use_mlflow is not None else settings.MLFLOW_ENABLED:
        artifacts["registry"] = log_to_mlflow(artifacts, {"data_source": "synthetic", "n_rows": n_rows, "seed": seed})
    return artifacts


if __name__ == "__main__":
    train()
