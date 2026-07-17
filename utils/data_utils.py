import numpy as np
import pandas as pd
import yfinance as yf
from typing import Optional

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

def get_observation_sequence(
    df: pd.DataFrame,
    col: str = "Returns",
) -> np.ndarray:
    return df[col].to_numpy(dtype=np.float64)

def time_split(df: pd.DataFrame, train_frac: float = 0.75):
    n = int(len(df) * train_frac)
    return df.iloc[:n].copy(), df.iloc[n:].copy()

def prepare_ticker_data(
    ticker: str = "SPY",
    start: str = "2010-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    df = fetch_ohlcv(ticker, start=start, end=end)
    df = add_log_returns(df)
    return df
