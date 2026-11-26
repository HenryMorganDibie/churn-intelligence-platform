"""
Application Configuration
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    APP_NAME: str = "Churn Intelligence Platform"
    APP_ENV: str = "development"
    DEBUG: bool = True

    # Database
    DATABASE_URL: str = "postgresql://churn_user:churn_pass@localhost:5432/churn_db"

    # Model
    MODEL_PATH: str = "models/churn_model.pkl"
    FEATURE_STORE_PATH: str = "models/feature_store.pkl"
    DRIFT_THRESHOLD: float = 0.15

    # MLflow tracking + registry
    MLFLOW_ENABLED: bool = True
    MLFLOW_TRACKING_URI: str = "file:./mlruns"
    MLFLOW_EXPERIMENT: str = "churn-intelligence"
    MLFLOW_MODEL_NAME: str = "churn-xgb"
    # When set (e.g. "champion"), the API loads the bundle of that registry alias
    # instead of MODEL_PATH. Empty = use the local file baked into the image.
    MLFLOW_SERVE_ALIAS: str = ""
    PROMOTION_MIN_AUC_GAIN: float = 0.002
    PROMOTION_MAX_BRIER_INCREASE: float = 0.002

    # Scoring
    CHURN_RISK_THRESHOLD: float = 0.5
    HIGH_RISK_THRESHOLD: float = 0.75

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
