"""Visualization helpers for regime detection outputs."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


REGIME_COLORS = {
    0: "rgba(46, 204, 113, 0.20)",
    1: "rgba(241, 196, 15, 0.20)",
    2: "rgba(231, 76, 60, 0.20)",
    3: "rgba(52, 152, 219, 0.20)",
    4: "rgba(155, 89, 182, 0.20)",
}


def plot_price_with_regimes(
    data: pd.DataFrame,
    state_col: str = "HiddenState",
    price_col: str = "Close",
    title: str = "Market Regimes",
) -> go.Figure:
    """Overlay hidden regimes on a price chart."""
    _require_columns(data, [state_col, price_col])
    figure = make_subplots(rows=1, cols=1)
    figure.add_trace(
        go.Scatter(
            x=data.index,
            y=data[price_col],
            mode="lines",
            name=price_col,
            line=dict(color="#1f2937", width=1.5),
        )
    )

    states = data[state_col].to_numpy()
    start = 0
    while start < len(states):
        end = start + 1
        while end < len(states) and states[end] == states[start]:
            end += 1
        state = int(states[start])
        figure.add_vrect(
            x0=data.index[start],
            x1=data.index[end - 1],
            fillcolor=REGIME_COLORS.get(state, "rgba(100, 116, 139, 0.16)"),
            line_width=0,
            layer="below",
        )
        start = end

    figure.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=price_col,
        template="plotly_white",
        hovermode="x unified",
    )
    return figure


def plot_transition_matrix(
    transition_probabilities: np.ndarray,
    labels: Optional[list[str]] = None,
    title: str = "Transition Probabilities",
) -> go.Figure:
    """Plot a transition probability heatmap."""
    matrix = np.asarray(transition_probabilities, dtype=float)
    labels = labels or [f"State {state}" for state in range(matrix.shape[0])]
    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            zmin=0,
            zmax=1,
            colorscale="Blues",
            text=np.round(matrix, 3),
            texttemplate="%{text}",
            hovertemplate="From %{y}<br>To %{x}<br>Probability %{z:.4f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=title,
        xaxis_title="Next State",
        yaxis_title="Current State",
        template="plotly_white",
    )
    return figure


def _require_columns(data: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
