"""
Churn Intelligence Platform — Main Application Entry Point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.routes import predict, health, batch
from app.core.config import settings
from app.core.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Churn Intelligence Platform...")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Churn Intelligence Platform",
    description="Real-time customer churn prediction with explainability and drift monitoring.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(predict.router, prefix="/predict", tags=["Prediction"])
app.include_router(batch.router, prefix="/batch", tags=["Batch Scoring"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
