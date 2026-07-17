"""Return preparation and input validation for regime models."""

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


def prepare_market_features(
    data: pd.DataFrame,
    price_col: str = "Close",
) -> pd.DataFrame:
    """Create a return frame from raw OHLCV data."""
    frame = data.copy()
    if "Returns" not in frame.columns:
        frame = add_log_returns(frame, price_col=price_col)
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
