"""
01_hmm_from_scratch/hmm_core.py
================================
From-scratch implementation of a Gaussian Hidden Markov Model.

Notation follows the Quant Guild reference PDF (Paolucci, 2025) exactly:

    λ = (π, A, µ, σ)
    N  — number of hidden states
    T  — length of observation sequence
    αt(i)  — forward variable     P(O₁..Ot, Qt=Si | λ)
    βt(i)  — backward variable    P(Ot+1..OT | Qt=Si, λ)
    γt(i)  — state occupancy      P(Qt=Si | O, λ)
    ξt(i,j) — transition occupancy P(Qt=Si, Qt+1=Sj | O, λ)

Three Canonical Problems (§3.2 of ref. PDF)
--------------------------------------------
P1  Evaluation  → forward_pass()  / log_likelihood()
P2  Decoding    → viterbi()
P3  Learning    → baum_welch()    (EM)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Gaussian emission helper
# ---------------------------------------------------------------------------

def _gaussian_pdf(x: float, mu: float, sigma: float) -> float:
    """b_j(O_t) = N(O_t; µ_j, σ_j²)   — scalar version."""
    return norm.pdf(x, loc=mu, scale=sigma)


def _emission_matrix(
    observations: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """
    Compute B ∈ ℝ^{T × N} where B[t, j] = b_j(O_t).

    Uses scipy.stats.norm for numerical stability.
    """
    T = len(observations)
    N = len(mu)
    B = np.zeros((T, N))
    for j in range(N):
        B[:, j] = norm.pdf(observations, loc=mu[j], scale=max(sigma[j], 1e-10))
    return B


# ---------------------------------------------------------------------------
# Problem 1a — Forward Algorithm
# ---------------------------------------------------------------------------

def forward_pass(
    observations: np.ndarray,
    pi: np.ndarray,
    A: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forward algorithm with scaling to avoid underflow.

    Parameters
    ----------
    observations : (T,) observed sequence O = (O₁, …, O_T)
    pi  : (N,)    initial state distribution  πᵢ = P(Q₁ = Sᵢ)
    A   : (N, N)  transition matrix           aᵢⱼ = P(Qt+1=Sj | Qt=Si)
    mu  : (N,)    emission means              µ_j
    sigma: (N,)   emission std devs           σ_j

    Returns
    -------
    alpha  : (T, N)  scaled forward variables  ᾱt(i)
    scales : (T,)    scaling factors           ct

    Notes
    -----
    The un-scaled recursion (Eq. 19 of ref. PDF):
        α₁(i)   = πᵢ · bᵢ(O₁)
        αt+1(j) = [Σᵢ αt(i) · aᵢⱼ] · bⱼ(Ot+1)

    Scaling: at each t we set c_t = Σᵢ ᾱt(i)  and divide through.
    Log-likelihood: ln P(O | λ) = −Σ_t ln c_t
    """
    T = len(observations)
    N = len(pi)
    B = _emission_matrix(observations, mu, sigma)

    alpha = np.zeros((T, N))
    scales = np.zeros(T)

    # --- Initialisation (Eq. 18) ---
    alpha[0] = pi * B[0]
    scales[0] = alpha[0].sum()
    if scales[0] < 1e-300:
        scales[0] = 1e-300
    alpha[0] /= scales[0]

    # --- Recursion (Eq. 19) ---
    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = np.dot(alpha[t - 1], A[:, j]) * B[t, j]
        scales[t] = alpha[t].sum()
        if scales[t] < 1e-300:
            scales[t] = 1e-300
        alpha[t] /= scales[t]

    return alpha, scales


# ---------------------------------------------------------------------------
# Problem 1b — Backward Algorithm
# ---------------------------------------------------------------------------

def backward_pass(
    observations: np.ndarray,
    A: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """
    Backward algorithm with scaling (uses the same c_t from forward pass).

    βt(i) = P(Ot+1, …, OT | Qt=Si, λ)

    Initialisation: βT(i) = 1        (Eq. 22)
    Recursion:      βt(i) = Σⱼ aᵢⱼ · bⱼ(Ot+1) · βt+1(j)  (Eq. 23)

    Returns
    -------
    beta : (T, N) scaled backward variables
    """
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


# ---------------------------------------------------------------------------
# State occupancies γ and ξ
# ---------------------------------------------------------------------------

def compute_gamma(
    alpha: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """
    State occupancy:
        γt(i) = P(Qt=Si | O, λ) = αt(i)·βt(i) / Σk αt(k)·βt(k)   (Eq. 25)

    Returns
    -------
    gamma : (T, N)
    """
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
    """
    Transition occupancy:
        ξt(i,j) = P(Qt=Si, Qt+1=Sj | O, λ)
               = αt(i) · aᵢⱼ · bⱼ(Ot+1) · βt+1(j)
                 ─────────────────────────────────────  (Eq. 26)
                 Σᵢ' Σⱼ' αt(i')·aᵢ'ⱼ'·bⱼ'(Ot+1)·βt+1(j')

    Returns
    -------
    xi : (T-1, N, N)
    """
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


# ---------------------------------------------------------------------------
# Problem 2 — Viterbi Algorithm
# ---------------------------------------------------------------------------

def viterbi(
    observations: np.ndarray,
    pi: np.ndarray,
    A: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Viterbi algorithm for the most-probable state sequence (Problem P2).

    δt(i) = max_{Q₁..Qt-1} P(Q₁,..,Qt=Si, O₁,..,Ot | λ)

    Recursion in log-space to avoid underflow:
        log δt(j) = max_i [log δt-1(i) + log aᵢⱼ] + log bⱼ(Ot)
        ψt(j)     = argmax_i [log δt-1(i) + log aᵢⱼ]

    Returns
    -------
    path     : (T,) most-probable hidden state sequence
    log_prob : log P(best path | O, λ)
    """
    T = len(observations)
    N = len(pi)
    B = _emission_matrix(observations, mu, sigma)

    log_A = np.log(np.maximum(A, 1e-300))
    log_B = np.log(np.maximum(B, 1e-300))
    log_pi = np.log(np.maximum(pi, 1e-300))

    # δ in log-space
    log_delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)

    # Initialisation
    log_delta[0] = log_pi + log_B[0]

    # Recursion
    for t in range(1, T):
        for j in range(N):
            trans_probs = log_delta[t - 1] + log_A[:, j]
            psi[t, j] = np.argmax(trans_probs)
            log_delta[t, j] = trans_probs[psi[t, j]] + log_B[t, j]

    # Termination
    log_prob = log_delta[T - 1].max()
    path = np.zeros(T, dtype=int)
    path[T - 1] = np.argmax(log_delta[T - 1])

    # Back-tracking
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]

    return path, log_prob


# ---------------------------------------------------------------------------
# Problem 3 — Baum-Welch (EM) Training
# ---------------------------------------------------------------------------

class GaussianHMM:
    """
    Gaussian Hidden Markov Model trained via the Baum-Welch algorithm.

    Parameters
    ----------
    n_states  : int   — number of hidden states N
    n_iter    : int   — maximum EM iterations
    tol       : float — convergence tolerance ε (Eq. 32 of ref. PDF)
    random_state : int or None

    Attributes (after fit)
    ----------------------
    pi_     : (N,)    initial distribution
    A_      : (N, N)  transition matrix
    mu_     : (N,)    emission means
    sigma_  : (N,)    emission std devs
    log_likelihoods_ : list of float — LL per EM iteration
    """

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

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _initialise(self, observations: np.ndarray):
        rng = np.random.default_rng(self.random_state)
        N = self.n_states

        # Uniform initial distribution
        self.pi_ = np.ones(N) / N

        # Uniform transition matrix (slightly perturbed for symmetry-breaking)
        A = rng.dirichlet(np.ones(N) * 5, size=N)
        self.A_ = A

        # k-means style mean initialisation via percentiles
        percentiles = np.linspace(10, 90, N)
        self.mu_ = np.percentile(observations, percentiles)

        # Equal variance initialisation
        self.sigma_ = np.full(N, observations.std())

        self.log_likelihoods_: List[float] = []

    # ------------------------------------------------------------------
    # Log-likelihood from scaled forward pass
    # ------------------------------------------------------------------

    @staticmethod
    def _log_likelihood_from_scales(scales: np.ndarray) -> float:
        """ln P(O | λ) = Σ_t ln c_t  (negative because c_t < 1 after normalisation)."""
        return np.sum(np.log(np.maximum(scales, 1e-300)))

    # ------------------------------------------------------------------
    # Fit (Baum-Welch EM — Algorithm 2 of ref. PDF)
    # ------------------------------------------------------------------

    def fit(self, observations: np.ndarray) -> "GaussianHMM":
        """
        Train the HMM using Baum-Welch EM on a 1-D observation sequence.

        Parameters
        ----------
        observations : (T,) float array

        Returns
        -------
        self
        """
        self._initialise(observations)
        T = len(observations)
        N = self.n_states

        prev_ll = -np.inf

        for iteration in range(self.n_iter):

            # ---- E-Step ------------------------------------------------
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

            # ---- M-Step ------------------------------------------------
            # Initial state (Eq. 27)
            self.pi_ = gamma[0]

            # Transition matrix (Eq. 28)
            xi_sum     = xi.sum(axis=0)            # (N, N)
            gamma_sum  = gamma[:-1].sum(axis=0)    # (N,)
            self.A_    = xi_sum / np.maximum(gamma_sum[:, None], 1e-300)
            # Row-normalise for numerical safety
            row_sums = self.A_.sum(axis=1, keepdims=True)
            self.A_ /= np.maximum(row_sums, 1e-300)

            # Emission parameters (Eqs. 29–30)
            gamma_sum_full = gamma.sum(axis=0)  # (N,)
            for j in range(N):
                denom = max(gamma_sum_full[j], 1e-300)
                self.mu_[j]    = (gamma[:, j] * observations).sum() / denom
                self.sigma_[j] = np.sqrt(
                    (gamma[:, j] * (observations - self.mu_[j]) ** 2).sum() / denom
                )
                self.sigma_[j] = max(self.sigma_[j], 1e-6)  # prevent collapse

            # ---- Convergence check (Eq. 32) ----------------------------
            ll = self._log_likelihood_from_scales(scales)
            self.log_likelihoods_.append(ll)

            if abs(ll - prev_ll) < self.tol:
                print(f"  Converged at iteration {iteration + 1}  (ΔLL = {abs(ll - prev_ll):.2e})")
                break
            prev_ll = ll

        else:
            print(f"  Reached max iterations ({self.n_iter}).")

        # Sort states by volatility (low → high) for interpretability
        order = np.argsort(self.sigma_)
        self.pi_    = self.pi_[order]
        self.A_     = self.A_[np.ix_(order, order)]
        self.mu_    = self.mu_[order]
        self.sigma_ = self.sigma_[order]

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_proba(self, observations: np.ndarray) -> np.ndarray:
        """Soft state assignments: γt(i) for each t.  Returns (T, N) array."""
        alpha, scales = forward_pass(
            observations, self.pi_, self.A_, self.mu_, self.sigma_
        )
        beta = backward_pass(
            observations, self.A_, self.mu_, self.sigma_, scales
        )
        return compute_gamma(alpha, beta)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Hard state assignments: argmax_i γt(i) for each t.  Returns (T,) array."""
        return np.argmax(self.predict_proba(observations), axis=1)

    def predict_viterbi(self, observations: np.ndarray) -> Tuple[np.ndarray, float]:
        """Most-probable state path via Viterbi decoding."""
        return viterbi(observations, self.pi_, self.A_, self.mu_, self.sigma_)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def log_likelihood(self, observations: np.ndarray) -> float:
        """Compute ln P(O | λ) on any observation sequence."""
        _, scales = forward_pass(
            observations, self.pi_, self.A_, self.mu_, self.sigma_
        )
        return self._log_likelihood_from_scales(scales)

    def score(self, observations: np.ndarray) -> float:
        """Alias for log_likelihood (per observation, normalised)."""
        return self.log_likelihood(observations) / len(observations)

    # ------------------------------------------------------------------
    # Forward & Backward accessors (for notebook visualisation)
    # ------------------------------------------------------------------

    def get_forward_backward(
        self, observations: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns alpha (T,N), beta (T,N), scales (T,) for the given sequence.
        """
        alpha, scales = forward_pass(
            observations, self.pi_, self.A_, self.mu_, self.sigma_
        )
        beta = backward_pass(
            observations, self.A_, self.mu_, self.sigma_, scales
        )
        return alpha, beta, scales

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

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
