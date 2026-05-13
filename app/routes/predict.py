"""Prediction routes."""

from fastapi import APIRouter, HTTPException, status

from app.schemas.predict import PredictionRequest, PredictionResponse
from app.services.regime_service import predict_regimes


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
