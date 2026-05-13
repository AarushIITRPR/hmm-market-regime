"""Application entry point."""

from fastapi import FastAPI

from app.routes.health import router as health_router
from app.routes.predict import router as predict_router


app = FastAPI(
    title="Market Regime Detection API",
    version="0.1.0",
)

app.include_router(health_router)
app.include_router(predict_router)
