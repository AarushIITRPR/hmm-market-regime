"""Service layer for regime prediction."""

from __future__ import annotations

import pandas as pd

from app.schemas.predict import OHLCVRecord, PredictionRequest, PredictionResponse
from market_regime import predict_market_regime


def predict_regimes(request: PredictionRequest) -> PredictionResponse:
    """Run market regime inference for a prediction request."""
    frame = _records_to_frame(request.data)
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
