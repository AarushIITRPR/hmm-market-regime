"""Data loading helpers for market regime detection."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf


def load_market_data(
    ticker: str,
    start: str = "2010-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance."""
    data = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
    )
    if data.empty:
        raise ValueError(f"No market data returned for ticker {ticker!r}.")

    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
    return data.dropna().sort_index()


def load_csv_data(
    path: str | Path,
    date_col: Optional[str] = None,
    index_col: Optional[str] = None,
) -> pd.DataFrame:
    """Load OHLCV or feature data from a CSV file."""
    data = pd.read_csv(path, parse_dates=[date_col] if date_col else None)
    if index_col:
        data = data.set_index(index_col)
    elif date_col:
        data = data.set_index(date_col)
    return data.dropna(how="all").sort_index()
