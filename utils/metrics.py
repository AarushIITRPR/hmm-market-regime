import numpy as np
import pandas as pd
from typing import Dict


def aic(log_likelihood: float, n_params: int) -> float:
    return 2 * n_params - 2 * log_likelihood


def bic(log_likelihood: float, n_params: int, n_obs: int) -> float:
    return n_params * np.log(n_obs) - 2 * log_likelihood


def hmm_n_params(N: int) -> int:
    return N ** 2 + 2 * N


def regime_persistence(labels: np.ndarray) -> Dict:
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


def conditional_sharpe(
    returns: np.ndarray,
    labels: np.ndarray,
    annualise: bool = True,
) -> Dict:
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
