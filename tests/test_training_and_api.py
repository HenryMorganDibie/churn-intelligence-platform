"""
Tests for training, the champion/challenger gate, MLflow registry serving and the API.
Run: pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.models import predictor
from app.models.trainer import fit_and_evaluate, generate_synthetic_data, promotion_gate, train
from tests.test_features import SAMPLE_CUSTOMER


# ── Promotion gate (pure logic) ──────────────────────────────────────────────

def test_gate_promotes_first_model():
    assert promotion_gate({"test_auc": 0.7, "test_brier": 0.2}, None, 0.002, 0.002).promoted


def test_gate_rejects_insufficient_auc_gain():
    d = promotion_gate({"test_auc": 0.801, "test_brier": 0.15}, {"test_auc": 0.800, "test_brier": 0.15}, 0.002, 0.002)
    assert not d.promoted and "AUC gain" in d.reason


def test_gate_rejects_worse_calibration_even_with_better_auc():
    d = promotion_gate({"test_auc": 0.85, "test_brier": 0.20}, {"test_auc": 0.80, "test_brier": 0.15}, 0.002, 0.002)
    assert not d.promoted and "Brier" in d.reason


def test_gate_promotes_clear_improvement():
    assert promotion_gate({"test_auc": 0.82, "test_brier": 0.149}, {"test_auc": 0.80, "test_brier": 0.15}, 0.002, 0.002).promoted


# ── Training: three disjoint splits, sane metrics ────────────────────────────

def test_fit_uses_disjoint_train_calib_test_splits():
    art = fit_and_evaluate(generate_synthetic_data(n=1500, seed=1), params={"n_estimators": 30})
    m = art["metrics"]
    assert m["n_train"] + m["n_calib"] + m["n_test"] == 1500
    assert m["n_calib"] > 0 and m["n_test"] > 0
    assert 0.5 < m["test_auc"] <= 1.0
    assert 0.0 <= m["test_brier"] < 0.25


# ── MLflow registry + API ────────────────────────────────────────────────────

@pytest.fixture()
def registry(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "MLFLOW_TRACKING_URI", f"file:{tmp_path / 'mlruns'}")
    monkeypatch.setattr(settings, "MODEL_PATH", str(tmp_path / "churn_model.pkl"))
    monkeypatch.setattr(settings, "MLFLOW_SERVE_ALIAS", "")
    predictor.load_artifacts.cache_clear()
    first = train(n_rows=1500, seed=1, params={"n_estimators": 60}, use_mlflow=True)
    yield first
    predictor.load_artifacts.cache_clear()


def test_registry_champion_then_rejected_challenger(registry):
    assert registry["registry"]["promoted"] is True
    assert str(registry["registry"]["version"]) == "1"
    weak = train(n_rows=1500, seed=1, params={"n_estimators": 2, "max_depth": 1}, use_mlflow=True)
    assert weak["registry"]["promoted"] is False
    assert weak["registry"]["previous_champion"]["version"] in ("1", 1)


def test_api_serves_champion_from_registry(registry, monkeypatch):
    monkeypatch.setattr(settings, "MLFLOW_SERVE_ALIAS", "champion")
    predictor.load_artifacts.cache_clear()
    from app.main import app

    client = TestClient(app)
    assert client.get("/health/").status_code == 200
    assert client.get("/health/ready").json() == {"ready": True}

    model = client.get("/health/model").json()
    assert model["provenance"]["source"] == "mlflow"
    assert model["provenance"]["alias"] == "champion"

    r = client.post("/predict/", json=SAMPLE_CUSTOMER)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_segment"] in {"low", "medium", "high", "critical"}
    assert body["top_risk_drivers"]


def test_readiness_is_503_without_a_model(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "MODEL_PATH", str(tmp_path / "missing.pkl"))
    monkeypatch.setattr(settings, "MLFLOW_SERVE_ALIAS", "")
    predictor.load_artifacts.cache_clear()
    from app.main import app

    r = TestClient(app).get("/health/ready")
    assert r.status_code == 503 and r.json()["ready"] is False
    predictor.load_artifacts.cache_clear()
