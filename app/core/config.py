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

    # Scoring
    CHURN_RISK_THRESHOLD: float = 0.5
    HIGH_RISK_THRESHOLD: float = 0.75

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
