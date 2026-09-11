"""Regimes -- MS-GARCH crisis-probability signal vs. realized volatility.

Per CLAUDE.md: the walk-forward regime probability barely correlates with
volatility (the 2-regime MS-GARCH doesn't identify in 500-obs windows).
Only `prob_crisis_insample` (a full-sample fit) tracks it, and it is a
high-vol-day detector, not a sustained-regime signal -- labelled as such here,
never re-plotted as if it were the walk-forward VaR driver.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cryptorisk.config import load_config
from cryptorisk.dashboard import data as D
from cryptorisk.dashboard.theme import AMBER, RED, TEXT, page_setup

page_setup("Regimes")
cfg = load_config()
st.title("Regimes")
st.caption(
    "MS-GARCH 2-regime crisis probability (full-sample fit, in-sample) as a "
    "high-volatility-day detector -- **not** a sustained walk-forward regime "
    "signal. The VaR/ES the decision layer uses is still walk-forward."
)

asset = st.selectbox("Asset", cfg["assets"])

try:
    con = duckdb.connect(D.store_path(), read_only=True)
    exists = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name='msgarch_predictions'"
    ).fetchone()[0]
    pred = (
        con.execute(
            "SELECT date, sigma2, prob_crisis_insample, prob_crisis_pred "
            "FROM msgarch_predictions WHERE asset = ? ORDER BY date",
            [asset],
        ).df()
        if exists
        else pd.DataFrame()
    )
    con.close()
except Exception:
    pred = pd.DataFrame()

if pred.empty:
    st.warning("No MS-GARCH predictions cached. Run `make msgarch` then `make regime-id`.")
else:
    pred["date"] = pd.to_datetime(pred["date"])
    win = D.load_price_window(asset, n=len(pred) + 30)
    merged = pred.merge(win[["date", "close", "log_return"]], on="date", how="left")
    merged["abs_ret_21d"] = merged["log_return"].abs().rolling(21).mean()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=merged["date"], y=merged["close"], name="close", line=dict(color=TEXT, width=1)
        )
    )
    fig.update_layout(height=280, title=f"{asset} price", yaxis_title="USD")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=merged["date"],
            y=merged["prob_crisis_insample"],
            name="P(crisis) -- in-sample",
            fill="tozeroy",
            line=dict(color=RED, width=1),
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=merged["date"],
            y=merged["abs_ret_21d"] / merged["abs_ret_21d"].max(),
            name="|return| 21d avg (scaled)",
            line=dict(color=AMBER, width=1, dash="dot"),
            yaxis="y2",
        )
    )
    fig2.update_layout(
        height=320,
        yaxis=dict(title="P(crisis regime)", range=[0, 1]),
        yaxis2=dict(overlaying="y", side="right", showgrid=False, title="scaled |ret|"),
        title="In-sample crisis probability vs. realized vol proxy",
    )
    st.plotly_chart(fig2, use_container_width=True)

    current = merged.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Current P(crisis), in-sample", f"{current['prob_crisis_insample']:.0%}")
    c2.metric(
        "Current P(crisis), walk-forward",
        f"{current['prob_crisis_pred']:.0%}",
        help="Near-zero OOS correlation with vol -- shown for completeness only.",
    )
    c3.metric(
        "MS-GARCH sigma (next-day)",
        f"{np.sqrt(current['sigma2']):.4f}" if pd.notna(current["sigma2"]) else "n/a",
    )

st.divider()
st.subheader("Regime-identification correlations (`make regime-id`)")
reg = D.load_results()["regime"]
r = reg[reg.asset == asset] if not reg.empty else pd.DataFrame()
if not r.empty:
    st.dataframe(
        r[
            [
                "series",
                "kind",
                "corr_absret",
                "spearman_absret",
                "corr_rv",
                "spearman_rv21",
                "mean_prob",
            ]
        ].rename(
            columns={
                "corr_absret": "corr(|ret|)",
                "spearman_absret": "spearman(|ret|)",
                "corr_rv": "corr(RV)",
                "spearman_rv21": "spearman(RV,21d)",
                "mean_prob": "mean P(crisis)",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "`insample` tracks vol (~0.5-0.75); `filt_wf`/`pred_wf` (walk-forward) barely do (~0.0-0.1) -- the MS-GARCH doesn't identify in 500-day rolling windows."
    )
