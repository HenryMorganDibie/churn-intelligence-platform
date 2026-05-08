# 📉 Churn Intelligence Platform

> **Real-time customer churn prediction with SHAP explainability, drift monitoring, and a production-ready REST API.**

Built as a reference implementation of a principal-level ML system — covering the full stack from feature engineering to serving, monitoring, and CI/CD.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-orange)
![CI](https://github.com/HenryMorganDibie/churn-intelligence-platform/actions/workflows/ci.yml/badge.svg)

---

🔴 **[Live Demo →](https://henrymorgandibie.github.io/churn-intelligence-platform)**


---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CLIENT / DASHBOARD                    │
│              Streamlit · REST API Consumer               │
└───────────────────────────┬─────────────────────────────┘
                            │ HTTP
┌───────────────────────────▼─────────────────────────────┐
│                     FastAPI Service                      │
│    /predict  ·  /batch/score  ·  /health  ·  /drift     │
└──────┬─────────────────────┬──────────────────┬─────────┘
       │                     │                  │
┌──────▼──────┐   ┌──────────▼──────┐  ┌───────▼────────┐
│   Feature   │   │  XGBoost Model  │  │ Drift Monitor  │
│   Store     │   │  + Calibration  │  │ (PSI-based)    │
│  Engineer   │   │  + SHAP Explain │  │                │
└─────────────┘   └─────────────────┘  └────────────────┘
                            │
              ┌─────────────▼──────────────┐
              │     PostgreSQL / TimescaleDB │
              │     Prediction Logs · Events │
              └─────────────────────────────┘
```

---

## ✨ Key Features

| Layer | What's built |
|---|---|
| **Feature Engineering** | Time-aware derived features: engagement score, loyalty score, payment risk score, support intensity |
| **Model** | XGBoost with isotonic probability calibration — reliable churn probabilities, not just scores |
| **Explainability** | SHAP TreeExplainer returns top 5 risk drivers per customer in real time |
| **Drift Detection** | Population Stability Index (PSI) across all features — flags distribution shifts before they hurt model performance |
| **Serving** | FastAPI with async support, single-customer and batch endpoints (up to 10K records) |
| **Monitoring** | Streamlit dashboard with live scoring, risk distribution charts, and batch simulation |
| **Infra** | Docker Compose, GitHub Actions CI, environment-variable config via `.env` |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- PostgreSQL (or use Docker Compose)

### Local Setup

```bash
# Clone
git clone https://github.com/HenryMorganDibie/churn-intelligence-platform.git
cd churn-intelligence-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your database credentials

# Train the model (generates models/churn_model.pkl)
python -m app.models.trainer

# Start the API
uvicorn app.main:app --reload
```

API docs available at **http://localhost:8000/docs**

### Run the Dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard available at **http://localhost:8501**

### Docker Compose (full stack)

```bash
docker-compose up --build
```

---

## 📡 API Reference

### `POST /predict/` — Score a Single Customer

```json
{
  "customer_id": "CUST-00123",
  "tenure_days": 180,
  "monthly_charges": 49.99,
  "total_charges": 899.82,
  "num_support_tickets": 3,
  "num_logins_last_30d": 12,
  "avg_session_duration_min": 8.5,
  "plan_changes_last_90d": 1,
  "payment_failures_last_6m": 1,
  "has_contract": false,
  "is_autopay": true,
  "product_tier": "standard"
}
```

**Response:**

```json
{
  "customer_id": "CUST-00123",
  "churn_probability": 0.6821,
  "risk_segment": "high",
  "top_risk_drivers": {
    "payment_risk_score": 0.412,
    "engagement_score": -0.287,
    "loyalty_score": -0.201,
    "num_support_tickets": 0.183,
    "tenure_days": -0.142
  },
  "recommended_action": "Trigger retention workflow. Assign CSM outreach.",
  "scored_at": "2025-01-15T10:32:00Z"
}
```

### `POST /batch/score` — Score up to 10,000 Customers

```json
{
  "customers": [ ... ]
}
```

### `GET /health/model` — Model Status

```json
{
  "model_loaded": true,
  "auc_roc": 0.8712,
  "brier_score": 0.1643
}
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Tests cover feature engineering edge cases, column alignment, encoding correctness, and boundary conditions.

---

## 📊 Model Performance

Trained and evaluated on 8,000 synthetic customer records (80/20 split, stratified):

| Metric | Score |
|---|---|
| AUC-ROC | ~0.87 |
| Brier Score | ~0.16 |
| Calibration | Isotonic (CV=3) |

> To train on your own data: replace `generate_synthetic_data()` in `app/models/trainer.py` with your actual data ingestion logic. Feature schema is defined in `app/features/engineer.py`.

---

## 🗂️ Project Structure

```
churn-intelligence-platform/
├── app/
│   ├── api/routes/          # FastAPI route handlers
│   ├── core/                # Config, schemas, database, logger
│   ├── features/            # Feature engineering module
│   ├── models/              # Trainer + predictor
│   ├── monitoring/          # Drift detection (PSI)
│   └── main.py
├── dashboard/               # Streamlit monitoring UI
├── tests/                   # pytest test suite
├── .github/workflows/       # GitHub Actions CI
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## 🔧 Extending This System

- **Add retraining triggers**: Wire `monitoring/drift.py` to a Prefect or Airflow task that retrains when PSI > threshold
- **Persistent logging**: Extend `core/database.py` with a `predictions` table using TimescaleDB for time-series queries
- **Async batch jobs**: Swap the sync batch endpoint for Celery + Redis for very large workloads
- **Authentication**: Add API key middleware in `app/core/` before production deployment

---

## 👤 Author

**Henry Dibie** — ML Systems Engineer  
[LinkedIn](https://linkedin.com/in/kinghenrymorgan) · [GitHub](https://github.com/HenryMorganDibie)

---

## 📄 License

MIT License — see `LICENSE` for details.
