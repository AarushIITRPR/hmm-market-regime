"""Inference API for market regime detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from market_regime.hmm import GaussianHMM
from market_regime.model_training import train_hmm_model
from market_regime.preprocessing import to_observation_sequence


@dataclass(frozen=True)
class RegimePrediction:
    """Predicted HMM states and human-readable regime labels."""

    hidden_states: np.ndarray
    transition_probabilities: np.ndarray
    predicted_regime_labels: np.ndarray
    state_probabilities: np.ndarray
    model: GaussianHMM


def predict_market_regime(
    data,
    model: Optional[GaussianHMM] = None,
    n_states: int = 3,
    n_iter: int = 200,
    tol: float = 1e-6,
    random_state: int | None = 42,
    feature_col: str = "Returns",
) -> RegimePrediction:
    """
    Predict market regimes from returns, OHLCV data, or a feature DataFrame.

    Returns hidden states, the learned transition probability matrix, and
    interpretable regime labels ordered by state volatility.
    """
    observations = to_observation_sequence(data, feature_col=feature_col)
    fitted_model = model or train_hmm_model(
        observations,
        n_states=n_states,
        n_iter=n_iter,
        tol=tol,
        random_state=random_state,
    ).model

    hidden_states, _ = fitted_model.predict_viterbi(observations)
    state_probabilities = fitted_model.predict_proba(observations)
    regime_names = label_regimes(observations, hidden_states, fitted_model.sigmas_)
    predicted_labels = np.array([regime_names[state] for state in hidden_states], dtype=object)

    return RegimePrediction(
        hidden_states=hidden_states,
        transition_probabilities=fitted_model.transition_matrix_.copy(),
        predicted_regime_labels=predicted_labels,
        state_probabilities=state_probabilities,
        model=fitted_model,
    )


def label_regimes(
    observations: np.ndarray,
    hidden_states: np.ndarray,
    state_volatility: np.ndarray,
) -> dict[int, str]:
    """Assign simple business labels using state volatility and average return."""
    labels: dict[int, str] = {}
    volatility_order = np.argsort(state_volatility)
    volatility_bucket = {
        volatility_order[0]: "Low Volatility",
        volatility_order[-1]: "High Volatility",
    }

    for state in sorted(set(hidden_states)):
        state_returns = observations[hidden_states == state]
        trend = "Bull" if state_returns.mean() >= 0 else "Bear"
        volatility = volatility_bucket.get(state, "Neutral Volatility")
        labels[int(state)] = f"{volatility} {trend}"

    return labels


def prediction_to_frame(
    data: pd.DataFrame,
    prediction: RegimePrediction,
    state_col: str = "HiddenState",
    label_col: str = "RegimeLabel",
) -> pd.DataFrame:
    """Attach regime predictions to a market DataFrame."""
    if len(data) != len(prediction.hidden_states):
        data = data.iloc[-len(prediction.hidden_states) :].copy()
    else:
        data = data.copy()
    data[state_col] = prediction.hidden_states
    data[label_col] = prediction.predicted_regime_labels
    return data
