"""Feature engineering and input validation for regime models."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_FEATURE_COLUMNS = ("Returns",)


def add_log_returns(data: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """Add a log-return column named ``Returns``."""
    _require_columns(data, [price_col])
    frame = data.copy()
    frame["Returns"] = np.log(frame[price_col] / frame[price_col].shift(1))
    return frame.dropna()


def add_realized_volatility(
    data: pd.DataFrame,
    windows: Sequence[int] = (20, 60),
    return_col: str = "Returns",
) -> pd.DataFrame:
    """Add annualized rolling volatility features."""
    _require_columns(data, [return_col])
    frame = data.copy()
    for window in windows:
        frame[f"RVol_{window}d"] = frame[return_col].rolling(window).std() * np.sqrt(252)
    return frame.dropna()


def add_technical_features(data: pd.DataFrame, return_col: str = "Returns") -> pd.DataFrame:
    """Add common return-derived features used in regime research."""
    _require_columns(data, [return_col])
    frame = data.copy()
    frame["AbsReturn"] = frame[return_col].abs()
    frame["ReturnSq"] = frame[return_col] ** 2
    frame["Momentum_5d"] = frame[return_col].rolling(5).sum()
    frame["RSkew_60d"] = frame[return_col].rolling(60).skew()
    frame["RKurt_60d"] = frame[return_col].rolling(60).kurt()

    if {"Open", "High", "Low", "Close"}.issubset(frame.columns):
        log_hl = np.log(frame["High"] / frame["Low"]) ** 2
        log_co = np.log(frame["Close"] / frame["Open"]) ** 2
        frame["GK_Vol"] = np.sqrt(252 * (0.5 * log_hl - (2 * np.log(2) - 1) * log_co))

    return frame.dropna()


def prepare_market_features(
    data: pd.DataFrame,
    price_col: str = "Close",
    volatility_windows: Sequence[int] = (20, 60),
    include_technical_features: bool = True,
) -> pd.DataFrame:
    """Create a production-ready feature frame from raw OHLCV data."""
    frame = data.copy()
    if "Returns" not in frame.columns:
        frame = add_log_returns(frame, price_col=price_col)
    frame = add_realized_volatility(frame, windows=volatility_windows)
    if include_technical_features:
        frame = add_technical_features(frame)
    return frame.dropna()


def to_observation_sequence(
    data: pd.DataFrame | pd.Series | np.ndarray | Iterable[float],
    feature_col: str = "Returns",
) -> np.ndarray:
    """Convert supported inputs into a clean univariate observation sequence."""
    if isinstance(data, pd.DataFrame):
        if feature_col not in data.columns:
            if "Close" in data.columns:
                data = prepare_market_features(data)
            else:
                raise ValueError(f"DataFrame must contain {feature_col!r} or a Close column.")
        values = data[feature_col].to_numpy(dtype=np.float64)
    elif isinstance(data, pd.Series):
        values = data.to_numpy(dtype=np.float64)
    else:
        values = np.asarray(list(data) if not isinstance(data, np.ndarray) else data, dtype=np.float64)

    values = values.reshape(-1)
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError("At least two finite observations are required.")
    return values


def _require_columns(data: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
