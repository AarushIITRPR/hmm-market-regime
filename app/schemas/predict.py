"""Request and response models for regime prediction."""

from __future__ import annotations

from datetime import date as Date
from typing import Optional

from pydantic import BaseModel, Field, model_validator


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
    n_iter: int = Field(default=100, ge=1, le=5000)
    tol: float = Field(default=1e-6, gt=0)
    random_state: Optional[int] = 42


class PredictionResponse(BaseModel):
    """Market regime prediction response."""

    hidden_states: list[int]
    transition_probabilities: list[list[float]]
    predicted_regime_labels: list[str]


class TickerPredictionRequest(BaseModel):
    """Payload for ticker-based market regime prediction."""

    ticker: str = Field(..., min_length=1, max_length=20, pattern=r"^[A-Za-z0-9.^=_-]+$")
    start_date: Date
    end_date: Optional[Date] = None
    n_states: int = Field(default=3, ge=2, le=10)
    n_iter: int = Field(default=100, ge=1, le=5000)
    tol: float = Field(default=1e-6, gt=0)
    random_state: Optional[int] = 42

    @model_validator(mode="after")
    def validate_date_range(self) -> "TickerPredictionRequest":
        """Ensure the requested date range is valid."""
        if self.end_date is not None and self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date.")
        return self


class TickerPredictionResponse(BaseModel):
    """Ticker-based market regime prediction response."""

    ticker: str
    dates: list[str]
    close_prices: list[float]
    returns: list[float]
    hidden_states: list[int]
    transition_probabilities: list[list[float]]
    predicted_regime_labels: list[str]
