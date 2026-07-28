"""Prediction routes."""

from fastapi import APIRouter, HTTPException, status

from app.schemas.predict import (
    TickerPredictionRequest,
    TickerPredictionResponse,
)
from app.services.regime_service import RegimeServiceError, predict_ticker_regimes


router = APIRouter(tags=["prediction"])


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
