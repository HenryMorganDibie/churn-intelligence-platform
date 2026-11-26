# 📉 Churn Intelligence Platform

> **Real-time customer churn prediction with SHAP explainability, drift monitoring, and a production-ready REST API.**

Built as a reference implementation of a principal-level ML system — covering the full stack from feature engineering to serving, monitoring, and CI/CD.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-orange)
![CI](https://github.com/HenryMorganDibie/churn-intelligence-platform/actions/workflows/ci.yml/badge.svg)

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
  "auc_roc": 0.986,
  "brier_score": 0.0467,
  "provenance": {"source": "mlflow", "model_name": "churn-xgb", "alias": "champion", "version": 1, "run_id": "..."}
}
```

`GET /health/ready` returns 200 once a model is loaded and 503 otherwise (used by the Kubernetes readiness probe).

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

17 tests: feature engineering edge cases and column alignment; disjoint train/calibration/test splits; the champion/challenger promotion gate; MLflow registration with a rejected weaker challenger; the API serving the registry `champion` alias end to end; and readiness returning 503 when no model is loaded.

---

## 📊 Model Performance

Measured on the held-out test split of 8,000 records (60% train / 20% calibration / 20% test, stratified; calibration and test are disjoint):

| Metric | Score |
|---|---|
| AUC-ROC | 0.986 |
| Brier Score (calibrated) | 0.047 |
| Brier Score (uncalibrated) | 0.048 |
| Calibration | Isotonic, fitted on the calibration split only |

> To train on your own data, swap the data loader in `app/models/trainer.py`; the feature schema is in `app/features/engineer.py`.

---

## 🧪 Experiment Tracking & Model Registry (MLflow)

Every training run is logged to MLflow: XGBoost parameters, data metadata, held-out AUC and Brier (calibrated and uncalibrated), the full artifact bundle (calibrated model, SHAP explainer, feature list, training reference sample) and an MLflow sklearn model with an inferred signature.

Each run registers a new version of `churn-xgb` and goes through a **champion/challenger gate**:

- the new version always gets the `challenger` alias;
- it takes the `champion` alias only if held-out AUC improves by at least `PROMOTION_MIN_AUC_GAIN` (default 0.002) **and** Brier does not worsen by more than `PROMOTION_MAX_BRIER_INCREASE` (default 0.002);
- the decision and reason are written as version tags (`gate_decision`, `gate_reason`).

```bash
python -m app.models.trainer                         # logs to ./mlruns
MLFLOW_SERVE_ALIAS=champion uvicorn app.main:app     # API loads whatever version is champion
curl localhost:8000/health/model                     # shows source, alias, version and run_id
```

---

## ☸️ Kubernetes Deployment

Kustomize manifests live in `deploy/k8s`:

| Path | What it deploys |
|---|---|
| `base/` | Namespace, ConfigMap, Deployment (2 replicas, rolling update with zero unavailable, startup/readiness/liveness probes, non-root, read-only root filesystem, dropped capabilities), Service, HPA (2 to 6 replicas at 70% CPU), PodDisruptionBudget |
| `overlays/dev/` | One replica, no HPA, for a laptop cluster |
| `overlays/mlflow/` | Adds an MLflow tracking server (SQLite + served artifacts on a PVC), a training Job that registers and gates a model, and points the API at the `champion` alias. API pods stay NotReady until the first champion exists, then become Ready |

The image is multi-stage: it trains a baseline model at build time, so the dev overlay works without MLflow, and runs as a non-root user.

```bash
make kind-up          # local cluster (kind)
make deploy-mlflow    # build, load, deploy MLflow + training Job + API
make smoke            # readiness, model provenance
make retrain          # run the Job again; the gate decides whether it becomes champion
make kind-down
```

CI (`.github/workflows/ci.yml`) runs the test suite, then a `k8s-e2e` job that builds the image, creates a kind cluster, deploys the MLflow overlay, waits for the training Job and API, smoke-tests prediction through the Service, and checks that a second training run is registered as v2 and gated.

SQLite and a single-replica MLflow server are demo choices. For shared use, back MLflow with Postgres and object storage (S3/GCS) and run it behind authentication.

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
├── deploy/k8s/              # Kustomize base + dev and MLflow overlays
├── .github/workflows/       # CI: tests + kind end-to-end deploy
├── Makefile
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## 🔧 Extending This System

- **Drift-triggered retraining**: Wire `monitoring/drift.py` to create the `churn-train` Job (or a Prefect/Airflow task) when PSI exceeds the threshold; the promotion gate already decides whether the result goes live
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
