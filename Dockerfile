# ── Stage 1: dependencies ───────────────────────────────────────────────────
FROM python:3.11-slim AS deps
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: train a baseline model into the image ──────────────────────────
# The image always ships a working model (MODEL_PATH), so pods are ready without
# MLflow. Set MLFLOW_SERVE_ALIAS at runtime to serve a registry alias instead.
FROM python:3.11-slim AS train
WORKDIR /app
COPY --from=deps /install /usr/local
COPY app ./app
RUN MLFLOW_ENABLED=false python -m app.models.trainer

# ── Stage 3: runtime (non-root) ─────────────────────────────────────────────
FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 APP_ENV=production DEBUG=false
WORKDIR /app
RUN useradd --uid 10001 --create-home appuser
COPY --from=deps /install /usr/local
COPY app ./app
COPY dashboard ./dashboard
COPY --from=train /app/models ./models
RUN chown -R appuser:appuser /app
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/ready').status==200 else 1)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
