"""Request and response models for regime prediction."""

from __future__ import annotations

from datetime import date as Date
from typing import Optional

from pydantic import BaseModel, Field


class OHLCVRecord(BaseModel):
    """Single OHLCV observation."""

    date: Optional[Date] = None
    open: float = Field(..., gt=0)
    high: float = Field(..., gt=0)
    low: float = Field(..., gt=0)
    close: float = Field(..., gt=0)
    volume: Optional[float] = Field(default=None, ge=0)


class PredictionRequest(BaseModel):
    """Payload for market regime prediction."""

    data: list[OHLCVRecord] = Field(..., min_length=2)
    n_states: int = Field(default=3, ge=2, le=10)
    n_iter: int = Field(default=200, ge=1, le=5000)
    tol: float = Field(default=1e-6, gt=0)
    random_state: Optional[int] = 42


class PredictionResponse(BaseModel):
    """Market regime prediction response."""

    hidden_states: list[int]
    transition_probabilities: list[list[float]]
    predicted_regime_labels: list[str]
