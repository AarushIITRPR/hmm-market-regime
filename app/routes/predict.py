"""Prediction routes."""

from fastapi import APIRouter, HTTPException, status

from app.schemas.predict import (
    PredictionRequest,
    PredictionResponse,
    TickerPredictionRequest,
    TickerPredictionResponse,
)
from app.services.regime_service import RegimeServiceError, predict_regimes, predict_ticker_regimes


router = APIRouter(tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict market regimes from OHLCV records."""
    try:
        return predict_regimes(request)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.post("/predict-ticker", response_model=TickerPredictionResponse)
def predict_ticker(request: TickerPredictionRequest) -> TickerPredictionResponse:
    """Predict market regimes for a downloaded ticker history."""
    try:
        return predict_ticker_regimes(request)
    except RegimeServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
