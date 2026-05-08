"""
Pydantic Schemas — Request & Response Models
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime


# ── Request ──────────────────────────────────────────────────────────────────

class CustomerEvent(BaseModel):
    customer_id: str = Field(..., description="Unique customer identifier")
    tenure_days: int = Field(..., ge=0, description="Days since activation")
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    num_support_tickets: int = Field(default=0, ge=0)
    num_logins_last_30d: int = Field(default=0, ge=0)
    avg_session_duration_min: float = Field(default=0.0, ge=0)
    plan_changes_last_90d: int = Field(default=0, ge=0)
    payment_failures_last_6m: int = Field(default=0, ge=0)
    has_contract: bool = Field(default=False)
    is_autopay: bool = Field(default=False)
    product_tier: str = Field(default="basic", description="basic | standard | premium")

    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "CUST-00123",
                "tenure_days": 180,
                "monthly_charges": 49.99,
                "total_charges": 899.82,
                "num_support_tickets": 3,
                "num_logins_last_30d": 12,
                "avg_session_duration_min": 8.5,
                "plan_changes_last_90d": 1,
                "payment_failures_last_6m": 1,
                "has_contract": False,
                "is_autopay": True,
                "product_tier": "standard"
            }
        }


class BatchRequest(BaseModel):
    customers: List[CustomerEvent]


# ── Response ──────────────────────────────────────────────────────────────────

class ChurnPrediction(BaseModel):
    customer_id: str
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    risk_segment: str = Field(..., description="low | medium | high | critical")
    top_risk_drivers: Dict[str, float] = Field(
        default_factory=dict,
        description="Top SHAP features driving churn risk"
    )
    recommended_action: str
    scored_at: datetime


class BatchResponse(BaseModel):
    total_scored: int
    high_risk_count: int
    predictions: List[ChurnPrediction]
    job_id: str
    completed_at: datetime


class DriftReport(BaseModel):
    checked_at: datetime
    drift_detected: bool
    drift_score: float
    drifted_features: List[str]
    recommendation: str
