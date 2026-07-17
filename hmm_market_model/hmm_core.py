from __future__ import annotations

import numpy as np
from scipy.stats import norm
from typing import List, Optional, Tuple

# Gaussian emission helper

def _gaussian_pdf(x: float, mu: float, sigma: float) -> float:
    return norm.pdf(x, loc=mu, scale=sigma)


def _emission_matrix(
    observations: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    T = len(observations)
    N = len(mu)
    B = np.zeros((T, N))
    for j in range(N):
        B[:, j] = norm.pdf(observations, loc=mu[j], scale=max(sigma[j], 1e-10))
    return B


# Forward Algorithm

def forward_pass(
    observations: np.ndarray,
    pi: np.ndarray,
    A: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    T = len(observations)
    N = len(pi)
    B = _emission_matrix(observations, mu, sigma)

    alpha = np.zeros((T, N))
    scales = np.zeros(T)

    alpha[0] = pi * B[0]
    scales[0] = alpha[0].sum()
    if scales[0] < 1e-300:
        scales[0] = 1e-300
    alpha[0] /= scales[0]

    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = np.dot(alpha[t - 1], A[:, j]) * B[t, j]
        scales[t] = alpha[t].sum()
        if scales[t] < 1e-300:
            scales[t] = 1e-300
        alpha[t] /= scales[t]

    return alpha, scales

# Backward Algorithm

def backward_pass(
    observations: np.ndarray,
    A: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    T = len(observations)
    N = len(mu)
    B = _emission_matrix(observations, mu, sigma)

    beta = np.zeros((T, N))
    beta[T - 1] = 1.0  # Initialisation (Eq. 22)
    beta[T - 1] /= scales[T - 1]

    for t in range(T - 2, -1, -1):
        for i in range(N):
            beta[t, i] = np.dot(A[i], B[t + 1] * beta[t + 1])
        beta[t] /= scales[t]

    return beta

# State occupancies calculation

def compute_gamma(
    alpha: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    return gamma


def compute_xi(
    observations: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    A: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    T, N = alpha.shape
    B = _emission_matrix(observations, mu, sigma)
    xi = np.zeros((T - 1, N, N))

    for t in range(T - 1):
        for i in range(N):
            for j in range(N):
                xi[t, i, j] = (
                    alpha[t, i] * A[i, j] * B[t + 1, j] * beta[t + 1, j]
                )
        denom = xi[t].sum()
        if denom > 0:
            xi[t] /= denom

    return xi

def viterbi(
    observations: np.ndarray,
    pi: np.ndarray,
    A: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> Tuple[np.ndarray, float]:
    T = len(observations)
    N = len(pi)
    B = _emission_matrix(observations, mu, sigma)

    log_A = np.log(np.maximum(A, 1e-300))
    log_B = np.log(np.maximum(B, 1e-300))
    log_pi = np.log(np.maximum(pi, 1e-300))

    log_delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)

    log_delta[0] = log_pi + log_B[0]

    for t in range(1, T):
        for j in range(N):
            trans_probs = log_delta[t - 1] + log_A[:, j]
            psi[t, j] = np.argmax(trans_probs)
            log_delta[t, j] = trans_probs[psi[t, j]] + log_B[t, j]

    log_prob = log_delta[T - 1].max()
    path = np.zeros(T, dtype=int)
    path[T - 1] = np.argmax(log_delta[T - 1])

    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]

    return path, log_prob


# Baum-Welch (EM) Training

class GaussianHMM:
    def __init__(
        self,
        n_states: int = 3,
        n_iter: int = 200,
        tol: float = 1e-6,
        random_state: Optional[int] = 42,
    ):
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state


    def _initialise(self, observations: np.ndarray):
        rng = np.random.default_rng(self.random_state)
        N = self.n_states

        self.pi_ = np.ones(N) / N
        A = rng.dirichlet(np.ones(N) * 5, size=N)
        self.A_ = A
        percentiles = np.linspace(10, 90, N)
        self.mu_ = np.percentile(observations, percentiles)
        self.sigma_ = np.full(N, observations.std())

        self.log_likelihoods_: List[float] = []

    @staticmethod
    def _log_likelihood_from_scales(scales: np.ndarray) -> float:
        """ln P(O | λ) = Σ_t ln c_t  (negative because c_t < 1 after normalisation)."""
        return np.sum(np.log(np.maximum(scales, 1e-300)))


    def fit(self, observations: np.ndarray) -> "GaussianHMM":
        self._initialise(observations)
        T = len(observations)
        N = self.n_states

        prev_ll = -np.inf

        for iteration in range(self.n_iter):

            # E - Step
            alpha, scales = forward_pass(
                observations, self.pi_, self.A_, self.mu_, self.sigma_
            )
            beta = backward_pass(
                observations, self.A_, self.mu_, self.sigma_, scales
            )

            gamma = compute_gamma(alpha, beta)          # (T, N)
            xi    = compute_xi(                         # (T-1, N, N)
                observations, alpha, beta,
                self.A_, self.mu_, self.sigma_,
            )

            # M - Step
            self.pi_ = gamma[0]

            xi_sum     = xi.sum(axis=0)            # (N, N)
            gamma_sum  = gamma[:-1].sum(axis=0)    # (N,)
            self.A_    = xi_sum / np.maximum(gamma_sum[:, None], 1e-300)
            row_sums = self.A_.sum(axis=1, keepdims=True)
            self.A_ /= np.maximum(row_sums, 1e-300)


            gamma_sum_full = gamma.sum(axis=0)
            for j in range(N):
                denom = max(gamma_sum_full[j], 1e-300)
                self.mu_[j]    = (gamma[:, j] * observations).sum() / denom
                self.sigma_[j] = np.sqrt(
                    (gamma[:, j] * (observations - self.mu_[j]) ** 2).sum() / denom
                )
                self.sigma_[j] = max(self.sigma_[j], 1e-6)

            ll = self._log_likelihood_from_scales(scales)
            self.log_likelihoods_.append(ll)

            if abs(ll - prev_ll) < self.tol:
                print(f"  Converged at iteration {iteration + 1}  (ΔLL = {abs(ll - prev_ll):.2e})")
                break
            prev_ll = ll

        else:
            print(f"  Reached max iterations ({self.n_iter}).")


        order = np.argsort(self.sigma_)
        self.pi_    = self.pi_[order]
        self.A_     = self.A_[np.ix_(order, order)]
        self.mu_    = self.mu_[order]
        self.sigma_ = self.sigma_[order]

        return self


    def predict_proba(self, observations: np.ndarray) -> np.ndarray:
        alpha, scales = forward_pass(
            observations, self.pi_, self.A_, self.mu_, self.sigma_
        )
        beta = backward_pass(
            observations, self.A_, self.mu_, self.sigma_, scales
        )
        return compute_gamma(alpha, beta)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(observations), axis=1)

    def predict_viterbi(self, observations: np.ndarray) -> Tuple[np.ndarray, float]:
        return viterbi(observations, self.pi_, self.A_, self.mu_, self.sigma_)


    def log_likelihood(self, observations: np.ndarray) -> float:
        _, scales = forward_pass(
            observations, self.pi_, self.A_, self.mu_, self.sigma_
        )
        return self._log_likelihood_from_scales(scales)

    def score(self, observations: np.ndarray) -> float:
        return self.log_likelihood(observations) / len(observations)


    def get_forward_backward(
        self, observations: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        alpha, scales = forward_pass(
            observations, self.pi_, self.A_, self.mu_, self.sigma_
        )
        beta = backward_pass(
            observations, self.A_, self.mu_, self.sigma_, scales
        )
        return alpha, beta, scales

    def summary(self) -> str:
        lines = [
            f"GaussianHMM  N={self.n_states}",
            f"  π  = {np.round(self.pi_, 4)}",
            "  A  =",
        ]
        for row in self.A_:
            lines.append(f"       {np.round(row, 4)}")
        lines.append(f"  µ  = {np.round(self.mu_, 6)}")
        lines.append(f"  σ  = {np.round(self.sigma_, 6)}")
        if self.log_likelihoods_:
            lines.append(f"  Final LL = {self.log_likelihoods_[-1]:.4f}")
        return "\n".join(lines)
