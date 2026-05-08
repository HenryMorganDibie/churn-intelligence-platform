"""
Drift Detection — Population Stability Index (PSI) based monitoring
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import List

from app.core.config import settings
from app.core.schemas import DriftReport
from app.core.logger import logger


def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Compute Population Stability Index between a reference and production distribution.
    PSI < 0.1: No drift | 0.1–0.2: Moderate drift | >0.2: Significant drift
    """
    def _bucket(arr, bins):
        counts, _ = np.histogram(arr, bins=bins)
        pct = counts / len(arr)
        return np.clip(pct, 1e-6, None)  # avoid log(0)

    bins = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    bins[0] -= 1e-6
    bins[-1] += 1e-6

    expected_pct = _bucket(expected, bins)
    actual_pct = _bucket(actual, bins)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return round(float(psi), 4)


def check_drift(
    reference_df: pd.DataFrame,
    incoming_df: pd.DataFrame,
    feature_columns: List[str],
) -> DriftReport:
    """
    Compare incoming batch distributions against reference (training) stats.
    Returns a DriftReport with per-feature PSI scores.
    """
    drifted_features = []
    max_psi = 0.0

    for col in feature_columns:
        if col not in reference_df.columns or col not in incoming_df.columns:
            continue
        psi = compute_psi(
            reference_df[col].dropna().values,
            incoming_df[col].dropna().values,
        )
        logger.debug(f"PSI [{col}]: {psi}")
        if psi > settings.DRIFT_THRESHOLD:
            drifted_features.append(col)
        max_psi = max(max_psi, psi)

    drift_detected = len(drifted_features) > 0

    if drift_detected:
        recommendation = (
            f"Drift detected in {len(drifted_features)} feature(s): "
            f"{', '.join(drifted_features)}. Consider retraining the model."
        )
    else:
        recommendation = "No significant drift detected. Model is stable."

    return DriftReport(
        checked_at=datetime.now(timezone.utc),
        drift_detected=drift_detected,
        drift_score=max_psi,
        drifted_features=drifted_features,
        recommendation=recommendation,
    )
