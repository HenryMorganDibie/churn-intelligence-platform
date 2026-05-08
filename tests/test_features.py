"""
Tests — Feature Engineering & Prediction Pipeline
Run: pytest tests/ -v
"""

import pytest
import pandas as pd
from app.features.engineer import engineer_features, engineer_batch_features, FEATURE_COLUMNS


SAMPLE_CUSTOMER = {
    "customer_id": "TEST-001",
    "tenure_days": 365,
    "monthly_charges": 49.99,
    "total_charges": 599.88,
    "num_support_tickets": 2,
    "num_logins_last_30d": 15,
    "avg_session_duration_min": 12.5,
    "plan_changes_last_90d": 1,
    "payment_failures_last_6m": 0,
    "has_contract": True,
    "is_autopay": True,
    "product_tier": "standard",
}


class TestFeatureEngineering:
    def test_output_is_dataframe(self):
        df = engineer_features(SAMPLE_CUSTOMER)
        assert isinstance(df, pd.DataFrame)

    def test_correct_columns(self):
        df = engineer_features(SAMPLE_CUSTOMER)
        assert list(df.columns) == FEATURE_COLUMNS

    def test_single_row(self):
        df = engineer_features(SAMPLE_CUSTOMER)
        assert len(df) == 1

    def test_no_nulls(self):
        df = engineer_features(SAMPLE_CUSTOMER)
        assert df.isnull().sum().sum() == 0

    def test_engagement_score_range(self):
        df = engineer_features(SAMPLE_CUSTOMER)
        assert 0.0 <= df["engagement_score"].iloc[0] <= 1.0

    def test_loyalty_score_range(self):
        df = engineer_features(SAMPLE_CUSTOMER)
        assert 0.0 <= df["loyalty_score"].iloc[0] <= 1.0

    def test_zero_tenure_guard(self):
        customer = {**SAMPLE_CUSTOMER, "tenure_days": 0}
        df = engineer_features(customer)
        assert df["tenure_days"].iloc[0] == 1  # clamped to 1

    def test_batch_engineering(self):
        records = [SAMPLE_CUSTOMER, {**SAMPLE_CUSTOMER, "customer_id": "TEST-002"}]
        df = engineer_batch_features(records)
        assert len(df) == 2
        assert list(df.columns) == FEATURE_COLUMNS

    def test_product_tier_encoding(self):
        for tier, expected in [("basic", 0), ("standard", 1), ("premium", 2)]:
            customer = {**SAMPLE_CUSTOMER, "product_tier": tier}
            df = engineer_features(customer)
            assert df["product_tier_encoded"].iloc[0] == expected
