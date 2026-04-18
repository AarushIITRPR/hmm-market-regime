import numpy as np
import pandas as pd
import yfinance as yf
from typing import Optional, Tuple

def fetch_ohlcv(
    ticker: str,
    start: str = "2010-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.dropna(inplace=True)
    return df

def add_log_returns(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    df = df.copy()
    df["Returns"] = np.log(df[price_col] / df[price_col].shift(1))
    df.dropna(inplace=True)
    return df

def add_realized_vol(
    df: pd.DataFrame,
    windows: Tuple[int, ...] = (20, 60),
    return_col: str = "Returns",
) -> pd.DataFrame:
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

def get_observation_sequence(
    df: pd.DataFrame,
    col: str = "Returns",
) -> np.ndarray:
    return df[col].to_numpy(dtype=np.float64)

def get_feature_matrix(
    df: pd.DataFrame,
    cols: list,
) -> np.ndarray:
    return df[cols].to_numpy(dtype=np.float64)

def empirical_transition_matrix(
    labels: np.ndarray,
    state_names: Optional[list] = None,
) -> Tuple[np.ndarray, list]:
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

def time_split(df: pd.DataFrame, train_frac: float = 0.75):
    n = int(len(df) * train_frac)
    return df.iloc[:n].copy(), df.iloc[n:].copy()

def prepare_ticker_data(
    ticker: str = "SPY",
    start: str = "2010-01-01",
    end: Optional[str] = None,
    rv_windows: Tuple[int, ...] = (20, 60),
) -> pd.DataFrame:
    df = fetch_ohlcv(ticker, start=start, end=end)
    df = add_log_returns(df)
    df = add_realized_vol(df, windows=rv_windows)
    df = add_regime_labels_from_vol(df)
    df = add_extra_features(df)
    return df