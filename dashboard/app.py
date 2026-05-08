"""
Churn Intelligence Platform — Monitoring Dashboard
Run: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Churn Intelligence Platform",
    page_icon="📉",
    layout="wide",
)

st.title("📉 Churn Intelligence Platform")
st.caption("Real-time churn risk monitoring · Powered by XGBoost + SHAP")

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Settings")
risk_threshold = st.sidebar.slider("High Risk Threshold", 0.0, 1.0, 0.55, 0.05)
st.sidebar.markdown("---")
st.sidebar.markdown("**API Status**")

try:
    r = requests.get(f"{API_BASE}/health/", timeout=3)
    if r.status_code == 200:
        st.sidebar.success("🟢 API Online")
    else:
        st.sidebar.error("🔴 API Offline")
except Exception:
    st.sidebar.error("🔴 API Offline")

# ── Score a Single Customer ────────────────────────────────────────────────────
st.header("🔍 Score a Customer")

with st.expander("Enter Customer Data", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        customer_id = st.text_input("Customer ID", "CUST-00123")
        tenure_days = st.number_input("Tenure (days)", 0, 3650, 180)
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 500.0, 49.99)
        total_charges = st.number_input("Total Charges ($)", 0.0, 50000.0, 899.82)
    with col2:
        num_support_tickets = st.number_input("Support Tickets", 0, 50, 3)
        num_logins_last_30d = st.number_input("Logins (last 30d)", 0, 200, 12)
        avg_session_duration_min = st.number_input("Avg Session (min)", 0.0, 300.0, 8.5)
    with col3:
        plan_changes_last_90d = st.number_input("Plan Changes (90d)", 0, 20, 1)
        payment_failures_last_6m = st.number_input("Payment Failures (6m)", 0, 20, 1)
        has_contract = st.checkbox("Has Contract", False)
        is_autopay = st.checkbox("On Autopay", True)
        product_tier = st.selectbox("Product Tier", ["basic", "standard", "premium"])

    if st.button("Score Customer", type="primary"):
        payload = {
            "customer_id": customer_id,
            "tenure_days": tenure_days,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "num_support_tickets": num_support_tickets,
            "num_logins_last_30d": num_logins_last_30d,
            "avg_session_duration_min": avg_session_duration_min,
            "plan_changes_last_90d": plan_changes_last_90d,
            "payment_failures_last_6m": payment_failures_last_6m,
            "has_contract": has_contract,
            "is_autopay": is_autopay,
            "product_tier": product_tier,
        }
        try:
            resp = requests.post(f"{API_BASE}/predict/", json=payload, timeout=10)
            result = resp.json()

            col_a, col_b, col_c = st.columns(3)
            prob = result.get("churn_probability", 0)
            segment = result.get("risk_segment", "unknown")
            color_map = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}

            col_a.metric("Churn Probability", f"{prob:.1%}")
            col_b.metric("Risk Segment", f"{color_map.get(segment, '')} {segment.upper()}")
            col_c.metric("Action", result.get("recommended_action", ""))

            if result.get("top_risk_drivers"):
                st.subheader("Top Risk Drivers (SHAP)")
                drivers = result["top_risk_drivers"]
                driver_df = pd.DataFrame(
                    list(drivers.items()), columns=["Feature", "SHAP Value"]
                ).sort_values("SHAP Value", key=abs, ascending=True)
                st.bar_chart(driver_df.set_index("Feature"))

        except Exception as e:
            st.error(f"API Error: {e}")

# ── Batch Simulation ────────────────────────────────────────────────────────────
st.header("📦 Batch Risk Simulation")
st.caption("Generate synthetic customers and view risk distribution")

if st.button("Run Simulation (100 customers)"):
    rng = np.random.default_rng(42)
    sim_customers = [
        {
            "customer_id": f"SIM-{i:04d}",
            "tenure_days": int(rng.integers(7, 1095)),
            "monthly_charges": float(round(rng.uniform(20, 150), 2)),
            "total_charges": float(round(rng.uniform(50, 5000), 2)),
            "num_support_tickets": int(rng.integers(0, 10)),
            "num_logins_last_30d": int(rng.integers(0, 45)),
            "avg_session_duration_min": float(round(rng.uniform(0, 60), 1)),
            "plan_changes_last_90d": int(rng.integers(0, 4)),
            "payment_failures_last_6m": int(rng.integers(0, 5)),
            "has_contract": bool(rng.integers(0, 2)),
            "is_autopay": bool(rng.integers(0, 2)),
            "product_tier": rng.choice(["basic", "standard", "premium"]),
        }
        for i in range(100)
    ]

    try:
        resp = requests.post(
            f"{API_BASE}/batch/score",
            json={"customers": sim_customers},
            timeout=30,
        )
        data = resp.json()
        predictions = data.get("predictions", [])
        df = pd.DataFrame(predictions)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Scored", data["total_scored"])
        col2.metric("High/Critical Risk", data["high_risk_count"])
        col3.metric(
            "High Risk Rate",
            f"{data['high_risk_count'] / data['total_scored']:.1%}"
        )
        col4.metric("Job ID", data["job_id"][:8] + "...")

        st.subheader("Risk Distribution")
        segment_counts = df["risk_segment"].value_counts()
        st.bar_chart(segment_counts)

        st.subheader("Churn Probability Distribution")
        st.area_chart(df["churn_probability"].sort_values().reset_index(drop=True))

        with st.expander("View All Predictions"):
            st.dataframe(
                df[["customer_id", "churn_probability", "risk_segment", "recommended_action"]],
                use_container_width=True,
            )

    except Exception as e:
        st.error(f"Simulation failed: {e}")
