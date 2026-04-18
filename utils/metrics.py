"""
utils/metrics.py
----------------
Evaluation metrics for regime detection models.

Metrics
-------
- AIC / BIC
- Log-likelihood
- Regime persistence (average duration per state)
- Regime purity (when ground-truth labels exist)
- Conditional Sharpe per regime
- Wasserstein distance between regime return distributions
"""

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Information criteria
# ---------------------------------------------------------------------------

def aic(log_likelihood: float, n_params: int) -> float:
    """AIC = 2k − 2L  (lower is better)."""
    return 2 * n_params - 2 * log_likelihood


def bic(log_likelihood: float, n_params: int, n_obs: int) -> float:
    """BIC = k·ln(T) − 2L  (lower is better)."""
    return n_params * np.log(n_obs) - 2 * log_likelihood


def hmm_n_params(N: int) -> int:
    """
    Free parameters for a Gaussian HMM with N states:
        N²  transition probs   (each row sums to 1 → N(N-1) free, but standard count is N²)
        N   means
        N   variances
        N-1 free initial state probs
    Simplified to N² + 2N as per §7.1 of ref. PDF.
    """
    return N ** 2 + 2 * N


# ---------------------------------------------------------------------------
# Regime persistence
# ---------------------------------------------------------------------------

def regime_persistence(labels: np.ndarray) -> Dict:
    """
    Average consecutive run length per regime (proxy for regime stability).

    Returns
    -------
    dict  {label: mean_duration_days}
    """
    durations: Dict = {}
    i = 0
    while i < len(labels):
        j = i + 1
        while j < len(labels) and labels[j] == labels[i]:
            j += 1
        lbl = labels[i]
        durations.setdefault(lbl, []).append(j - i)
        i = j
    return {lbl: np.mean(v) for lbl, v in durations.items()}


def regime_persistence_summary(labels: np.ndarray) -> pd.DataFrame:
    """Return a tidy DataFrame of persistence statistics per state."""
    durations: Dict = {}
    i = 0
    while i < len(labels):
        j = i + 1
        while j < len(labels) and labels[j] == labels[i]:
            j += 1
        lbl = labels[i]
        durations.setdefault(lbl, []).append(j - i)
        i = j
    rows = []
    for lbl, durs in durations.items():
        rows.append({
            "State": lbl,
            "Mean Duration (days)": np.mean(durs),
            "Median Duration (days)": np.median(durs),
            "Max Duration (days)": np.max(durs),
            "# Episodes": len(durs),
        })
    return pd.DataFrame(rows).sort_values("State").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Conditional Sharpe ratio per regime
# ---------------------------------------------------------------------------

def conditional_sharpe(
    returns: np.ndarray,
    labels: np.ndarray,
    annualise: bool = True,
) -> Dict:
    """
    Sharpe = (mean return / std return) per regime, optionally annualised (×√252).

    Returns
    -------
    dict  {label: sharpe_ratio}
    """
    factor = np.sqrt(252) if annualise else 1.0
    result = {}
    for lbl in set(labels):
        mask = labels == lbl
        r = returns[mask]
        if r.std() > 0:
            result[lbl] = (r.mean() / r.std()) * factor
        else:
            result[lbl] = np.nan
    return result


# ---------------------------------------------------------------------------
# Conditional statistics per regime
# ---------------------------------------------------------------------------

def regime_statistics(
    returns: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for lbl in sorted(set(labels), key=str):
        mask = labels == lbl
        r = returns[mask]
        rows.append({
            "State": lbl,
            "Mean Return (ann.)": r.mean() * 252,
            "Volatility (ann.)": r.std() * np.sqrt(252),
            "Sharpe (ann.)": (r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan,
            "Skewness": pd.Series(r).skew(),
            "Excess Kurtosis": pd.Series(r).kurt(),
            "# Days": mask.sum(),
            "% of Sample": mask.mean() * 100,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Wasserstein distance between regimes
# ---------------------------------------------------------------------------

def pairwise_wasserstein(
    returns: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    """
    Pairwise Wasserstein-1 distances between regime return distributions.
    """
    unique = sorted(set(labels), key=str)
    N = len(unique)
    W = np.zeros((N, N))
    for i, a in enumerate(unique):
        for j, b in enumerate(unique):
            if i != j:
                W[i, j] = wasserstein_distance(returns[labels == a], returns[labels == b])
    return pd.DataFrame(W, index=unique, columns=unique)


# ---------------------------------------------------------------------------
# Regime purity (vs an observable ground truth, e.g. vol buckets)
# ---------------------------------------------------------------------------

def regime_purity(pred_labels: np.ndarray, true_labels: np.ndarray) -> float:
    """
    Clustering purity: for each predicted cluster, count the majority true label.

    purity = (1/N) Σ_k max_j |pred==k ∩ true==j|
    """
    total, n = 0, len(pred_labels)
    for lbl in set(pred_labels):
        mask = pred_labels == lbl
        if mask.sum() == 0:
            continue
        counts = np.bincount(pd.Categorical(true_labels[mask]).codes)
        total += counts.max()
    return total / n


# ---------------------------------------------------------------------------
# Summary table helper
# ---------------------------------------------------------------------------

def model_comparison_table(results: List[Dict]) -> pd.DataFrame:
    """
    results: list of dicts with keys:
        model, n_states, log_likelihood, n_params, n_obs,
        mean_persistence, purity (optional)
    """
    rows = []
    for r in results:
        ll = r.get("log_likelihood", np.nan)
        k  = r.get("n_params", np.nan)
        T  = r.get("n_obs", np.nan)
        rows.append({
            "Model":             r.get("model", "?"),
            "States":            r.get("n_states", "?"),
            "Log-Likelihood":    round(ll, 2) if not np.isnan(ll) else "—",
            "AIC":               round(aic(ll, k), 1) if not np.isnan(ll) else "—",
            "BIC":               round(bic(ll, k, T), 1) if not np.isnan(ll) else "—",
            "Mean Persistence":  round(r.get("mean_persistence", np.nan), 1),
            "Purity vs Vol":     round(r.get("purity", np.nan), 3),
        })
    return pd.DataFrame(rows)
