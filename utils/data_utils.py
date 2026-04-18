"""
utils/data_utils.py
-------------------
Data fetching, preprocessing, and feature engineering for market-regime-detection.

Notation follows the Quant Guild reference PDF (Paolucci, 2025):
  r_t = ln(P_t / P_{t-1})   — log-return
  σ̂_t                        — rolling realized volatility (annualised)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_ohlcv(
    ticker: str,
    start: str = "2010-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance.

    Returns
    -------
    pd.DataFrame with columns: Open, High, Low, Close, Volume
    """
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.dropna(inplace=True)
    return df


# ---------------------------------------------------------------------------
# Return & volatility features
# ---------------------------------------------------------------------------

def add_log_returns(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """
    Append log-return series r_t = ln(P_t / P_{t-1}).
    """
    df = df.copy()
    df["Returns"] = np.log(df[price_col] / df[price_col].shift(1))
    df.dropna(inplace=True)
    return df


def add_realized_vol(
    df: pd.DataFrame,
    windows: Tuple[int, ...] = (20, 60),
    return_col: str = "Returns",
) -> pd.DataFrame:
    """
    Append rolling realised volatility (annualised):
        σ̂^(w)_t = √252 · std_{rolling w}(r_t)

    Parameters
    ----------
    windows : tuple of ints — rolling window sizes (days)
    """
    df = df.copy()
    for w in windows:
        df[f"RVol_{w}d"] = df[return_col].rolling(w).std() * np.sqrt(252)
    df.dropna(inplace=True)
    return df


def add_regime_labels_from_vol(
    df: pd.DataFrame,
    vol_col: str = "RVol_20d",
    low_q: float = 0.33,
    high_q: float = 0.66,
) -> pd.DataFrame:
    """
    Observable-factor regime labels via quantile bucketing (§2.2.1 of ref. PDF).

    Regime:
        Low  ← vol ≤ Q_{low_q}
        Med  ← Q_{low_q} < vol ≤ Q_{high_q}
        High ← vol > Q_{high_q}
    """
    df = df.copy()
    q_low = df[vol_col].quantile(low_q)
    q_high = df[vol_col].quantile(high_q)

    def _label(v):
        if v <= q_low:
            return "Low"
        elif v <= q_high:
            return "Med"
        else:
            return "High"

    df["VolRegime"] = df[vol_col].apply(_label)
    return df


def add_extra_features(df: pd.DataFrame, return_col: str = "Returns") -> pd.DataFrame:
    """
    Additional features used by alternative methods (§3):

    - Realised variance (5-day)
    - Realised skewness (60-day)
    - Realised kurtosis (60-day)
    - Absolute return (proxy for daily realised vol)
    - Return sign momentum (5-day sum of sign)
    - Garman-Klass volatility estimator (uses OHLC)
    """
    df = df.copy()
    df["AbsReturn"] = df[return_col].abs()
    df["ReturnSq"] = df[return_col] ** 2

    df["RSkew_60d"] = df[return_col].rolling(60).skew()
    df["RKurt_60d"] = df[return_col].rolling(60).kurt()

    df["Momentum_5d"] = df[return_col].rolling(5).sum()

    if all(c in df.columns for c in ["Open", "High", "Low", "Close"]):
        log_hl = np.log(df["High"] / df["Low"]) ** 2
        log_co = np.log(df["Close"] / df["Open"]) ** 2
        df["GK_Vol"] = np.sqrt(252 * (0.5 * log_hl - (2 * np.log(2) - 1) * log_co))

    df.dropna(inplace=True)
    return df


# ---------------------------------------------------------------------------
# Observation sequence helpers
# ---------------------------------------------------------------------------

def get_observation_sequence(
    df: pd.DataFrame,
    col: str = "Returns",
) -> np.ndarray:
    """Return 1-D float64 array ready for HMM training."""
    return df[col].to_numpy(dtype=np.float64)


def get_feature_matrix(
    df: pd.DataFrame,
    cols: list,
) -> np.ndarray:
    """Return (T, d) float64 matrix for multi-variate methods."""
    return df[cols].to_numpy(dtype=np.float64)


# ---------------------------------------------------------------------------
# Regime transition matrix from observable labels
# ---------------------------------------------------------------------------

def empirical_transition_matrix(
    labels: np.ndarray,
    state_names: Optional[list] = None,
) -> Tuple[np.ndarray, list]:
    """
    MLE transition matrix from a sequence of discrete labels.

    â_{ij} = n_{ij} / Σ_k n_{ik}        (Eq. 13 in ref. PDF)

    Returns
    -------
    A : (N, N) ndarray
    state_names : list of unique label strings
    """
    if state_names is None:
        state_names = sorted(set(labels))
    idx = {s: i for i, s in enumerate(state_names)}
    N = len(state_names)
    counts = np.zeros((N, N))
    for t in range(len(labels) - 1):
        counts[idx[labels[t]], idx[labels[t + 1]]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    A = counts / row_sums
    return A, state_names


# ---------------------------------------------------------------------------
# Train / test split (no shuffling — preserve time order)
# ---------------------------------------------------------------------------

def time_split(df: pd.DataFrame, train_frac: float = 0.75):
    n = int(len(df) * train_frac)
    return df.iloc[:n].copy(), df.iloc[n:].copy()


# ---------------------------------------------------------------------------
# Convenience wrapper: full pipeline for a single ticker
# ---------------------------------------------------------------------------

def prepare_ticker_data(
    ticker: str = "SPY",
    start: str = "2010-01-01",
    end: Optional[str] = None,
    rv_windows: Tuple[int, ...] = (20, 60),
) -> pd.DataFrame:
    """
    One-shot pipeline: download → log returns → realised vol → extra features → labels.
    """
    df = fetch_ohlcv(ticker, start=start, end=end)
    df = add_log_returns(df)
    df = add_realized_vol(df, windows=rv_windows)
    df = add_regime_labels_from_vol(df)
    df = add_extra_features(df)
    return df
