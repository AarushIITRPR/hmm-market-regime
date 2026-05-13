"""Service layer for regime prediction."""

from __future__ import annotations

import pandas as pd
import yfinance as yf
from fastapi import status

from app.schemas.predict import (
    OHLCVRecord,
    PredictionRequest,
    PredictionResponse,
    TickerPredictionRequest,
    TickerPredictionResponse,
)
from market_regime import predict_market_regime
from market_regime.preprocessing import add_log_returns


class RegimeServiceError(Exception):
    """Application error that can be mapped to an HTTP response."""

    def __init__(self, message: str, status_code: int = status.HTTP_400_BAD_REQUEST):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def predict_regimes(request: PredictionRequest) -> PredictionResponse:
    """Run market regime inference for a prediction request."""
    frame = _records_to_frame(request.data)
    frame = add_log_returns(frame)
    prediction = predict_market_regime(
        frame,
        n_states=request.n_states,
        n_iter=request.n_iter,
        tol=request.tol,
        random_state=request.random_state,
    )

    return PredictionResponse(
        hidden_states=prediction.hidden_states.astype(int).tolist(),
        transition_probabilities=prediction.transition_probabilities.tolist(),
        predicted_regime_labels=[str(label) for label in prediction.predicted_regime_labels],
    )


def predict_ticker_regimes(request: TickerPredictionRequest) -> TickerPredictionResponse:
    """Fetch historical ticker data and run market regime inference."""
    raw_data = _download_ticker_data(request)
    feature_frame = add_log_returns(raw_data)
    if feature_frame.empty:
        raise RegimeServiceError(
            "Downloaded data did not contain enough rows to compute returns.",
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    prediction = predict_market_regime(
        feature_frame,
        n_states=request.n_states,
        n_iter=request.n_iter,
        tol=request.tol,
        random_state=request.random_state,
    )

    return TickerPredictionResponse(
        ticker=request.ticker.upper(),
        dates=[_format_date(index_value) for index_value in feature_frame.index],
        close_prices=feature_frame["Close"].astype(float).tolist(),
        returns=feature_frame["Returns"].astype(float).tolist(),
        hidden_states=prediction.hidden_states.astype(int).tolist(),
        transition_probabilities=prediction.transition_probabilities.tolist(),
        predicted_regime_labels=[str(label) for label in prediction.predicted_regime_labels],
    )


def _records_to_frame(records: list[OHLCVRecord]) -> pd.DataFrame:
    rows = [
        {
            "Date": record.date,
            "Open": record.open,
            "High": record.high,
            "Low": record.low,
            "Close": record.close,
            "Volume": record.volume,
        }
        for record in records
    ]
    frame = pd.DataFrame(rows)
    if frame["Date"].notna().any():
        frame = frame.set_index(pd.to_datetime(frame["Date"])).drop(columns=["Date"])
    else:
        frame = frame.drop(columns=["Date"])
    return frame.dropna(axis=1, how="all")


def _download_ticker_data(request: TickerPredictionRequest) -> pd.DataFrame:
    try:
        data = yf.download(
            request.ticker,
            start=request.start_date.isoformat(),
            end=request.end_date.isoformat() if request.end_date else None,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        raise RegimeServiceError(
            f"Failed to download data for ticker {request.ticker!r}.",
            status.HTTP_502_BAD_GATEWAY,
        ) from exc

    if data is None or data.empty:
        raise RegimeServiceError(
            f"No historical data found for ticker {request.ticker!r}.",
            status.HTTP_404_NOT_FOUND,
        )

    data = data.copy()
    data.columns = [column[0] if isinstance(column, tuple) else column for column in data.columns]
    if "Close" not in data.columns:
        raise RegimeServiceError(
            f"Downloaded data for ticker {request.ticker!r} does not include close prices.",
            status.HTTP_502_BAD_GATEWAY,
        )

    return data.dropna(subset=["Close"]).sort_index()


def _format_date(index_value) -> str:
    timestamp = pd.Timestamp(index_value)
    return timestamp.date().isoformat()
