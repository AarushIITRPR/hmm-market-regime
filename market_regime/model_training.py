"""Model training utilities for HMM market regime detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from market_regime.hmm import GaussianHMM
from market_regime.preprocessing import to_observation_sequence


@dataclass(frozen=True)
class HMMTrainingResult:
    """Container returned by ``train_hmm_model``."""

    model: GaussianHMM
    observations: np.ndarray
    log_likelihood: float


def train_hmm_model(
    data,
    n_states: int = 3,
    n_iter: int = 200,
    tol: float = 1e-6,
    random_state: int | None = 42,
    feature_col: str = "Returns",
) -> HMMTrainingResult:
    """Fit a univariate Gaussian HMM to market observations."""
    observations = to_observation_sequence(data, feature_col=feature_col)
    model = GaussianHMM(
        n_states=n_states,
        n_iter=n_iter,
        tol=tol,
        random_state=random_state,
    ).fit(observations)
    return HMMTrainingResult(
        model=model,
        observations=observations,
        log_likelihood=model.log_likelihood(observations),
    )
