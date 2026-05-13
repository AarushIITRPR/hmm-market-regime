"""Numerically stable univariate Gaussian Hidden Markov Model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.stats import norm


def _as_observations(values: np.ndarray) -> np.ndarray:
    observations = np.asarray(values, dtype=np.float64).reshape(-1)
    if observations.size < 2:
        raise ValueError("At least two observations are required to fit an HMM.")
    if not np.isfinite(observations).all():
        raise ValueError("Observations contain NaN or infinite values.")
    return observations


def _emission_matrix(observations: np.ndarray, means: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    emissions = np.zeros((len(observations), len(means)))
    for state in range(len(means)):
        emissions[:, state] = norm.pdf(
            observations,
            loc=means[state],
            scale=max(sigmas[state], 1e-10),
        )
    return emissions


def forward_pass(
    observations: np.ndarray,
    initial_probs: np.ndarray,
    transition_matrix: np.ndarray,
    means: np.ndarray,
    sigmas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute scaled forward probabilities."""
    observations = _as_observations(observations)
    n_states = len(initial_probs)
    emissions = _emission_matrix(observations, means, sigmas)

    alpha = np.zeros((len(observations), n_states))
    scales = np.zeros(len(observations))

    alpha[0] = initial_probs * emissions[0]
    scales[0] = max(alpha[0].sum(), 1e-300)
    alpha[0] /= scales[0]

    for timestamp in range(1, len(observations)):
        for state in range(n_states):
            alpha[timestamp, state] = (
                np.dot(alpha[timestamp - 1], transition_matrix[:, state])
                * emissions[timestamp, state]
            )
        scales[timestamp] = max(alpha[timestamp].sum(), 1e-300)
        alpha[timestamp] /= scales[timestamp]

    return alpha, scales


def backward_pass(
    observations: np.ndarray,
    transition_matrix: np.ndarray,
    means: np.ndarray,
    sigmas: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """Compute scaled backward probabilities."""
    observations = _as_observations(observations)
    emissions = _emission_matrix(observations, means, sigmas)
    beta = np.zeros((len(observations), len(means)))
    beta[-1] = 1.0 / max(scales[-1], 1e-300)

    for timestamp in range(len(observations) - 2, -1, -1):
        for state in range(len(means)):
            beta[timestamp, state] = np.dot(
                transition_matrix[state],
                emissions[timestamp + 1] * beta[timestamp + 1],
            )
        beta[timestamp] /= max(scales[timestamp], 1e-300)

    return beta


def compute_gamma(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Return posterior state probabilities."""
    gamma = alpha * beta
    gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)
    return gamma


def compute_xi(
    observations: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    transition_matrix: np.ndarray,
    means: np.ndarray,
    sigmas: np.ndarray,
) -> np.ndarray:
    """Return expected transition counts for each timestamp."""
    observations = _as_observations(observations)
    emissions = _emission_matrix(observations, means, sigmas)
    n_periods, n_states = alpha.shape
    xi = np.zeros((n_periods - 1, n_states, n_states))

    for timestamp in range(n_periods - 1):
        for from_state in range(n_states):
            for to_state in range(n_states):
                xi[timestamp, from_state, to_state] = (
                    alpha[timestamp, from_state]
                    * transition_matrix[from_state, to_state]
                    * emissions[timestamp + 1, to_state]
                    * beta[timestamp + 1, to_state]
                )
        xi[timestamp] /= max(xi[timestamp].sum(), 1e-300)

    return xi


def viterbi(
    observations: np.ndarray,
    initial_probs: np.ndarray,
    transition_matrix: np.ndarray,
    means: np.ndarray,
    sigmas: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Decode the most likely hidden-state path."""
    observations = _as_observations(observations)
    emissions = _emission_matrix(observations, means, sigmas)
    n_periods = len(observations)
    n_states = len(initial_probs)

    log_transition = np.log(np.maximum(transition_matrix, 1e-300))
    log_emissions = np.log(np.maximum(emissions, 1e-300))
    log_initial = np.log(np.maximum(initial_probs, 1e-300))

    log_delta = np.zeros((n_periods, n_states))
    backpointers = np.zeros((n_periods, n_states), dtype=int)
    log_delta[0] = log_initial + log_emissions[0]

    for timestamp in range(1, n_periods):
        for state in range(n_states):
            transition_scores = log_delta[timestamp - 1] + log_transition[:, state]
            backpointers[timestamp, state] = int(np.argmax(transition_scores))
            log_delta[timestamp, state] = transition_scores[backpointers[timestamp, state]] + log_emissions[timestamp, state]

    path = np.zeros(n_periods, dtype=int)
    path[-1] = int(np.argmax(log_delta[-1]))
    for timestamp in range(n_periods - 2, -1, -1):
        path[timestamp] = backpointers[timestamp + 1, path[timestamp + 1]]

    return path, float(log_delta[-1].max())


@dataclass
class GaussianHMM:
    """Univariate Gaussian HMM trained with Baum-Welch EM."""

    n_states: int = 3
    n_iter: int = 200
    tol: float = 1e-6
    random_state: Optional[int] = 42
    initial_probs_: np.ndarray = field(init=False, repr=False)
    transition_matrix_: np.ndarray = field(init=False, repr=False)
    means_: np.ndarray = field(init=False, repr=False)
    sigmas_: np.ndarray = field(init=False, repr=False)
    log_likelihoods_: list[float] = field(default_factory=list, init=False)

    def fit(self, observations: np.ndarray) -> "GaussianHMM":
        observations = _as_observations(observations)
        self._initialise(observations)
        previous_log_likelihood = -np.inf

        for _ in range(self.n_iter):
            alpha, scales = forward_pass(
                observations,
                self.initial_probs_,
                self.transition_matrix_,
                self.means_,
                self.sigmas_,
            )
            beta = backward_pass(
                observations,
                self.transition_matrix_,
                self.means_,
                self.sigmas_,
                scales,
            )
            gamma = compute_gamma(alpha, beta)
            xi = compute_xi(
                observations,
                alpha,
                beta,
                self.transition_matrix_,
                self.means_,
                self.sigmas_,
            )

            self.initial_probs_ = gamma[0]
            self.transition_matrix_ = xi.sum(axis=0) / np.maximum(
                gamma[:-1].sum(axis=0)[:, None],
                1e-300,
            )
            self.transition_matrix_ /= np.maximum(
                self.transition_matrix_.sum(axis=1, keepdims=True),
                1e-300,
            )

            gamma_sum = gamma.sum(axis=0)
            for state in range(self.n_states):
                denominator = max(gamma_sum[state], 1e-300)
                self.means_[state] = (gamma[:, state] * observations).sum() / denominator
                variance = (gamma[:, state] * (observations - self.means_[state]) ** 2).sum()
                self.sigmas_[state] = max(np.sqrt(variance / denominator), 1e-6)

            log_likelihood = self._log_likelihood_from_scales(scales)
            self.log_likelihoods_.append(log_likelihood)
            if abs(log_likelihood - previous_log_likelihood) < self.tol:
                break
            previous_log_likelihood = log_likelihood

        self._sort_states_by_volatility()
        return self

    def predict(self, observations: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(observations), axis=1)

    def predict_proba(self, observations: np.ndarray) -> np.ndarray:
        alpha, scales = forward_pass(
            observations,
            self.initial_probs_,
            self.transition_matrix_,
            self.means_,
            self.sigmas_,
        )
        beta = backward_pass(
            observations,
            self.transition_matrix_,
            self.means_,
            self.sigmas_,
            scales,
        )
        return compute_gamma(alpha, beta)

    def predict_viterbi(self, observations: np.ndarray) -> tuple[np.ndarray, float]:
        return viterbi(
            observations,
            self.initial_probs_,
            self.transition_matrix_,
            self.means_,
            self.sigmas_,
        )

    def log_likelihood(self, observations: np.ndarray) -> float:
        _, scales = forward_pass(
            observations,
            self.initial_probs_,
            self.transition_matrix_,
            self.means_,
            self.sigmas_,
        )
        return self._log_likelihood_from_scales(scales)

    @staticmethod
    def _log_likelihood_from_scales(scales: np.ndarray) -> float:
        return float(np.sum(np.log(np.maximum(scales, 1e-300))))

    def _initialise(self, observations: np.ndarray) -> None:
        rng = np.random.default_rng(self.random_state)
        self.initial_probs_ = np.ones(self.n_states) / self.n_states
        self.transition_matrix_ = rng.dirichlet(np.ones(self.n_states) * 5, size=self.n_states)
        self.means_ = np.percentile(observations, np.linspace(10, 90, self.n_states))
        self.sigmas_ = np.full(self.n_states, max(observations.std(), 1e-6))
        self.log_likelihoods_ = []

    def _sort_states_by_volatility(self) -> None:
        order = np.argsort(self.sigmas_)
        self.initial_probs_ = self.initial_probs_[order]
        self.transition_matrix_ = self.transition_matrix_[np.ix_(order, order)]
        self.means_ = self.means_[order]
        self.sigmas_ = self.sigmas_[order]
