"""Plotly figures shared by several pages."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from stock_dashboard.logic.chart_helpers import RS_MA_LONG, RS_MA_SHORT

RS_CHART_HEIGHT = 350
# A horizontal legend above the plot's top-right corner, clear of the x-axis labels.
LEGEND_ABOVE = {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1}


def rs_figure(metrics: pd.DataFrame, title: str) -> go.Figure:
    """Relative strength with its 150- and 252-day moving averages.

    ``metrics`` needs date, relative_strength and the ``MA150``/``MA252`` columns
    of :func:`stock_dashboard.logic.chart_helpers.compute_rs_moving_averages`.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=metrics["date"], y=metrics["relative_strength"], name="Relative Strength")
    )
    for column, label, color in (
        (f"MA{RS_MA_SHORT}", f"{RS_MA_SHORT}-day MA", "orange"),
        (f"MA{RS_MA_LONG}", f"{RS_MA_LONG}-day MA", "red"),
    ):
        if column in metrics.columns:
            fig.add_trace(
                go.Scatter(
                    x=metrics["date"],
                    y=metrics[column],
                    name=label,
                    line={"dash": "dash", "color": color},
                )
            )
    fig.update_layout(
        title=title,
        yaxis_title="Relative Strength",
        height=RS_CHART_HEIGHT,
        legend=LEGEND_ABOVE,
    )
    return fig
