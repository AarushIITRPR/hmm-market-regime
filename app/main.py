"""Application entry point."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.health import router as health_router
from app.routes.optimize import router as optimize_router
from app.routes.predict import router as predict_router
from app.services.ray_service import initialize_ray, shutdown_ray


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_ray()
    yield
    shutdown_ray()


app = FastAPI(
    title="Market Regime Detection API",
    version="0.1.0",
    lifespan=lifespan,
)

cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "BACKEND_CORS_ORIGINS",
        "http://127.0.0.1:5173,http://localhost:5173,http://127.0.0.1:3000,http://localhost:3000",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(predict_router)
app.include_router(optimize_router)
