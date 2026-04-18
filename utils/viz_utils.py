"""
utils/viz_utils.py
------------------
Reusable interactive Plotly visualisation helpers for market regime detection.

All functions return plotly.graph_objects.Figure objects so callers can
further customise before calling .show().
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import norm
from typing import List, Optional, Dict


# ---------------------------------------------------------------------------
# Colour palette (consistent across notebooks)
# ---------------------------------------------------------------------------

REGIME_COLOURS = {
    0: "rgba(99,  200, 180, 0.85)",   # teal   – low vol / calm
    1: "rgba(255, 195,  55, 0.85)",   # amber  – medium vol
    2: "rgba(240,  80,  80, 0.85)",   # red    – high vol / stress
    3: "rgba(140, 100, 240, 0.85)",   # purple – 4th state
    4: "rgba( 80, 160, 240, 0.85)",   # blue   – 5th state
    "Low":  "rgba(99,  200, 180, 0.85)",
    "Med":  "rgba(255, 195,  55, 0.85)",
    "High": "rgba(240,  80,  80, 0.85)",
}

_DARK_LAYOUT = dict(
    plot_bgcolor="rgba(15,17,22,1)",
    paper_bgcolor="rgba(15,17,22,1)",
    font=dict(color="#e0e0e0", family="Inter, Arial, sans-serif"),
    xaxis=dict(showgrid=True, gridcolor="rgba(80,80,80,0.3)", zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="rgba(80,80,80,0.3)", zeroline=False),
    legend=dict(bgcolor="rgba(30,30,40,0.8)", bordercolor="rgba(80,80,80,0.5)", borderwidth=1),
)


def _apply_dark(fig: go.Figure) -> go.Figure:
    fig.update_layout(**_DARK_LAYOUT)
    return fig


# ---------------------------------------------------------------------------
# 1.  Price + regime background shading
# ---------------------------------------------------------------------------

def plot_price_with_regimes(
    df: pd.DataFrame,
    regime_col: str,
    price_col: str = "Close",
    title: str = "Price with Regime Overlay",
    regime_labels: Optional[Dict] = None,
) -> go.Figure:
    """
    Candlestick/line price chart with coloured background bands per regime.

    Parameters
    ----------
    df           : must have DatetimeIndex, price_col, regime_col
    regime_col   : column of integer or string regime labels
    regime_labels: optional dict {label: display_name}
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3],
                        subplot_titles=(title, "Log-Returns"))

    # Price line
    fig.add_trace(go.Scatter(
        x=df.index, y=df[price_col],
        name=price_col,
        line=dict(color="rgba(180,200,255,0.9)", width=1.2),
        hovertemplate="%{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>",
    ), row=1, col=1)

    # Returns bars
    if "Returns" in df.columns:
        colours = ["rgba(99,200,140,0.7)" if r >= 0 else "rgba(240,80,80,0.7)"
                   for r in df["Returns"]]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Returns"],
            name="Log-Return",
            marker_color=colours,
            hovertemplate="%{x|%Y-%m-%d}<br>Return: %{y:.4f}<extra></extra>",
        ), row=2, col=1)

    # Regime shading
    labels = df[regime_col].values
    unique = list(dict.fromkeys(labels))
    legend_added = set()
    i = 0
    while i < len(labels):
        j = i + 1
        while j < len(labels) and labels[j] == labels[i]:
            j += 1
        lbl = labels[i]
        colour = REGIME_COLOURS.get(lbl, "rgba(150,150,150,0.2)")
        show = lbl not in legend_added
        display = (regime_labels or {}).get(lbl, str(lbl))
        for row in [1, 2]:
            fig.add_vrect(
                x0=df.index[i], x1=df.index[j - 1],
                fillcolor=colour.replace("0.85", "0.18"),
                line_width=0,
                row=row, col=1,
                annotation_text="" if not show else "",
            )
        if show:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=12, color=colour, symbol="square"),
                name=display,
                showlegend=True,
            ))
            legend_added.add(lbl)
        i = j

    _apply_dark(fig)
    fig.update_layout(height=600, hovermode="x unified",
                      xaxis2_rangeslider_visible=False)
    return fig


# ---------------------------------------------------------------------------
# 2.  Regime-conditional return distributions
# ---------------------------------------------------------------------------

def plot_regime_distributions(
    df: pd.DataFrame,
    regime_col: str,
    return_col: str = "Returns",
    title: str = "Regime-Conditional Return Distributions",
) -> go.Figure:
    """Overlay empirical histogram + fitted Gaussian per regime."""
    fig = go.Figure()
    labels = sorted(df[regime_col].unique())
    x = np.linspace(df[return_col].quantile(0.001), df[return_col].quantile(0.999), 800)

    for lbl in labels:
        mask = df[regime_col] == lbl
        returns = df.loc[mask, return_col].dropna()
        mu, sigma = returns.mean(), returns.std()
        colour = REGIME_COLOURS.get(lbl, "rgba(200,200,200,0.7)")

        # Histogram
        fig.add_trace(go.Histogram(
            x=returns * 100,
            histnorm="probability density",
            name=f"State {lbl} (hist)",
            marker_color=colour.replace("0.85", "0.35"),
            xbins=dict(size=0.15),
            showlegend=True,
            hovertemplate=f"State {lbl}<br>Return: %{{x:.2f}}%<br>Density: %{{y:.2f}}<extra></extra>",
        ))

        # Fitted Gaussian PDF
        pdf = norm.pdf(x, mu, sigma)
        fig.add_trace(go.Scatter(
            x=x * 100, y=pdf,
            mode="lines",
            name=f"State {lbl} ~ N({mu*100:.2f}%, {sigma*100:.2f}%)",
            line=dict(color=colour, width=2.5),
        ))

    fig.update_layout(
        title=title, xaxis_title="Daily Return (%)", yaxis_title="Probability Density",
        barmode="overlay", height=480,
        **_DARK_LAYOUT,
    )
    return fig


# ---------------------------------------------------------------------------
# 3.  Transition matrix heatmap
# ---------------------------------------------------------------------------

def plot_transition_matrix(
    A: np.ndarray,
    state_names: Optional[List[str]] = None,
    title: str = "Learned Transition Matrix",
) -> go.Figure:
    N = A.shape[0]
    if state_names is None:
        state_names = [f"S{i}" for i in range(N)]

    fig = go.Figure(go.Heatmap(
        z=A,
        x=[f"→ {s}" for s in state_names],
        y=[f"{s} →" for s in state_names],
        colorscale="Teal",
        zmin=0, zmax=1,
        text=np.round(A, 3),
        texttemplate="%{text}",
        textfont=dict(size=14),
        hovertemplate="From %{y}<br>To %{x}<br>Prob: %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(title=title, height=400, **_DARK_LAYOUT)
    return fig


# ---------------------------------------------------------------------------
# 4.  Forward / Backward variable heatmap (diagnostic)
# ---------------------------------------------------------------------------

def plot_alpha_beta(
    alpha: np.ndarray,
    beta: np.ndarray,
    max_t: int = 200,
    state_names: Optional[List[str]] = None,
) -> go.Figure:
    N = alpha.shape[1]
    if state_names is None:
        state_names = [f"S{i}" for i in range(N)]

    T_plot = min(max_t, alpha.shape[0])
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("Forward Variables α_t(i)", "Backward Variables β_t(i)"),
                        shared_xaxes=True)

    for i in range(N):
        fig.add_trace(go.Scatter(
            x=list(range(T_plot)), y=alpha[:T_plot, i],
            name=f"α {state_names[i]}", mode="lines",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=list(range(T_plot)), y=beta[:T_plot, i],
            name=f"β {state_names[i]}", mode="lines", showlegend=True,
        ), row=2, col=1)

    fig.update_layout(title="Forward & Backward Variables", height=500, **_DARK_LAYOUT)
    return fig


# ---------------------------------------------------------------------------
# 5.  State occupancy (gamma) over time
# ---------------------------------------------------------------------------

def plot_gamma(
    gamma: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
    state_names: Optional[List[str]] = None,
    title: str = "State Occupancy Probabilities γ_t(i)",
) -> go.Figure:
    N = gamma.shape[1]
    if state_names is None:
        state_names = [f"State {i}" for i in range(N)]
    x = dates if dates is not None else list(range(len(gamma)))

    fig = go.Figure()
    for i in range(N):
        colour = REGIME_COLOURS.get(i, "rgba(200,200,200,0.8)")
        fig.add_trace(go.Scatter(
            x=x, y=gamma[:, i],
            name=state_names[i],
            stackgroup="one",
            line=dict(color=colour),
            fillcolor=colour.replace("0.85", "0.6"),
            hovertemplate="%{x|%Y-%m-%d}<br>" + state_names[i] + ": %{y:.3f}<extra></extra>",
        ))

    fig.update_layout(
        title=title,
        yaxis_title="P(Q_t = S_i | O, λ)",
        height=350, **_DARK_LAYOUT,
    )
    return fig


# ---------------------------------------------------------------------------
# 6.  Log-likelihood convergence curve
# ---------------------------------------------------------------------------

def plot_log_likelihood(
    log_likelihoods: List[float],
    title: str = "Baum-Welch Convergence — Log-Likelihood",
) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=list(range(1, len(log_likelihoods) + 1)),
        y=log_likelihoods,
        mode="lines+markers",
        line=dict(color="rgba(99,200,180,1)", width=2),
        marker=dict(size=5),
        hovertemplate="Iteration %{x}<br>LL: %{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="EM Iteration",
        yaxis_title="Log-Likelihood",
        height=380, **_DARK_LAYOUT,
    )
    return fig


# ---------------------------------------------------------------------------
# 7.  AIC / BIC comparison bar chart
# ---------------------------------------------------------------------------

def plot_aic_bic(
    model_names: List[str],
    aic_values: List[float],
    bic_values: List[float],
    title: str = "Model Selection: AIC vs BIC",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="AIC",
        x=model_names,
        y=aic_values,
        marker_color="rgba(99,200,180,0.8)",
    ))
    fig.add_trace(go.Bar(
        name="BIC",
        x=model_names,
        y=bic_values,
        marker_color="rgba(255,195,55,0.8)",
    ))
    fig.update_layout(
        title=title, barmode="group",
        yaxis_title="Score (lower = better)",
        height=400, **_DARK_LAYOUT,
    )
    return fig


# ---------------------------------------------------------------------------
# 8.  Viterbi path overlay on returns
# ---------------------------------------------------------------------------

def plot_viterbi_path(
    df: pd.DataFrame,
    viterbi_states: np.ndarray,
    return_col: str = "Returns",
    state_names: Optional[List[str]] = None,
    title: str = "Viterbi Most-Probable State Sequence",
) -> go.Figure:
    N = len(set(viterbi_states))
    if state_names is None:
        state_names = [f"State {i}" for i in range(N)]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4],
                        subplot_titles=(title, "Hidden State"))

    fig.add_trace(go.Scatter(
        x=df.index, y=df[return_col] * 100,
        name="Log-Return (%)",
        line=dict(color="rgba(180,200,255,0.6)", width=0.8),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=viterbi_states,
        name="Viterbi State",
        mode="lines",
        line=dict(color="rgba(255,195,55,0.9)", width=1.5),
        hovertemplate="%{x|%Y-%m-%d}<br>State: %{y}<extra></extra>",
    ), row=2, col=1)

    fig.update_yaxes(
        tickvals=list(range(N)),
        ticktext=state_names,
        row=2, col=1,
    )
    fig.update_layout(height=520, **_DARK_LAYOUT)
    return fig


# ---------------------------------------------------------------------------
# 9.  Regime statistics summary (returns & vol per state)
# ---------------------------------------------------------------------------

def plot_regime_stats(
    regime_stats: Dict,
    title: str = "Regime Statistics",
) -> go.Figure:
    """
    regime_stats: {label: {'mean': float, 'std': float, 'count': int}}
    """
    labels = list(regime_stats.keys())
    means  = [regime_stats[l]["mean"] * 100 for l in labels]
    stds   = [regime_stats[l]["std"] * 100 for l in labels]
    counts = [regime_stats[l]["count"] for l in labels]

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=("Mean Daily Return (%)", "Daily Volatility (%)", "# Days"))

    colours = [REGIME_COLOURS.get(l, "rgba(200,200,200,0.7)") for l in labels]

    fig.add_trace(go.Bar(x=[str(l) for l in labels], y=means,
                         marker_color=colours, name="Mean"), row=1, col=1)
    fig.add_trace(go.Bar(x=[str(l) for l in labels], y=stds,
                         marker_color=colours, name="Std"), row=1, col=2)
    fig.add_trace(go.Bar(x=[str(l) for l in labels], y=counts,
                         marker_color=colours, name="Count"), row=1, col=3)

    fig.update_layout(title=title, showlegend=False, height=380, **_DARK_LAYOUT)
    return fig
