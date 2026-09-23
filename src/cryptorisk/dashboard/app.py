"""Cryptorisk Terminal -- Overview page (live price + VaR/ES band).

``streamlit run src/cryptorisk/dashboard/app.py`` (or ``make dashboard``).
Streamlit auto-discovers ``pages/*.py`` next to this file for the other tabs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cryptorisk.config import load_config
from cryptorisk.dashboard import data as D
from cryptorisk.dashboard.theme import AMBER, GREEN, RED, TEXT, page_setup

page_setup("Overview")
cfg = load_config()
assets = list(cfg["assets"])
alphas = list(cfg["alphas"])

st.title("◈ Cryptorisk Terminal")
st.caption(
    "Live spot price vs. the study's out-of-sample VaR/ES band. "
    "Backtest numbers are frozen at the last pipeline run; the band's model "
    "is re-fit on demand for today's forecast only -- see the badges below."
)

top = st.columns([1, 1, 3])
asset = top[0].selectbox("Asset", assets, key="ov_asset")
alpha = top[1].selectbox(
    "Confidence", alphas, format_func=lambda a: f"{100 * (1 - a):.1f}%", key="ov_alpha"
)

results = D.load_results()
primary = D.primary_model(asset, alpha, results)
model_choices = D.model_names()
default_idx = model_choices.index(primary) if primary in model_choices else 0
model_name = top[2].selectbox(
    "Model (defaults to the study's FZ0-best, in-MCS model)",
    model_choices,
    index=default_idx,
    key="ov_model",
)

price = D.live_price(asset)
win = D.load_price_window(asset, n=260)
fc = D.today_forecast(asset, model_name, alphas=tuple(alphas))

# --------------------------------------------------------------------- #
# Ticker row
# --------------------------------------------------------------------- #
c1, c2, c3, c4 = st.columns(4)
if price:
    c1.metric(
        f"{asset}-USD (Binance spot)",
        f"${price['price']:,.2f}",
        f"{price['change_pct']:+.2f}% 24h",
    )
    c2.metric("24h High / Low", f"${price['high']:,.0f} / ${price['low']:,.0f}")
else:
    c1.metric(f"{asset}-USD", "n/a")
    c2.warning("Live price feed unavailable (network/API).")

if fc is not None:
    lo = fc.get(f"var_{alpha}")
    hi = fc.get(f"upper_{alpha}")
    last = fc["last_close"]
    lo_px = last * np.exp(lo) if lo is not None and np.isfinite(lo) else np.nan
    hi_px = last * np.exp(hi) if hi is not None and np.isfinite(hi) else np.nan
    label = f"Implied {100 * (1 - alpha):.1f}% range (next close)"
    if np.isfinite(hi_px):
        c3.metric(label, f"${lo_px:,.0f} – ${hi_px:,.0f}")
    else:
        c3.metric(
            f"VaR floor ({100 * (1 - alpha):.1f}%)",
            f"${lo_px:,.0f}",
            help="Model has no upper-tail quantile (e.g. CAViaR).",
        )
    es = fc.get(f"es_{alpha}")
    es_px = last * np.exp(es) if es is not None and np.isfinite(es) else np.nan
    c4.metric(
        f"ES floor ({100 * (1 - alpha):.1f}%)", f"${es_px:,.0f}" if np.isfinite(es_px) else "n/a"
    )
    st.markdown(
        f'<span class="cubo-badge cubo-badge-live">LIVE re-fit</span>'
        f"&nbsp;&nbsp;{model_name} refit on the latest 500 obs as of "
        f"{pd.Timestamp(fc['asof']).date()}, forecasting the next close. "
        f"Not the study's frozen backtest number.",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<span class="cubo-badge cubo-badge-frozen">FROZEN backtest</span>'
        f"&nbsp;&nbsp;Live re-fit for {model_name} unavailable on the current "
        "cached window (too short, or the fit raised); showing the last "
        "computed out-of-sample row instead.",
        unsafe_allow_html=True,
    )
    bt = D.load_backtests()
    row = bt[(bt.asset == asset) & (bt.model == model_name) & (bt.alpha == alpha)].tail(1)
    if not row.empty:
        r = row.iloc[0]
        c3.metric("Last VaR (log-return)", f"{r['var']:.4f}")
        c4.metric("Last ES (log-return)", f"{r['es']:.4f}")

st.divider()

# --------------------------------------------------------------------- #
# Price chart with the VaR/ES cone over the trailing window
# --------------------------------------------------------------------- #
bt = D.load_backtests()
sub = bt[(bt.asset == asset) & (bt.model == model_name) & (bt.alpha == alpha)].copy()
sub["date"] = pd.to_datetime(sub["date"])
hist = win.tail(180).copy()

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=hist["date"], y=hist["close"], name=f"{asset} close", line=dict(color=TEXT, width=1.6)
    )
)
if not sub.empty:
    band = sub.merge(hist[["date", "close"]], on="date", how="inner").sort_values("date")
    var_px = band["close"].shift(1).fillna(band["close"]) * np.exp(band["var"])
    es_px = band["close"].shift(1).fillna(band["close"]) * np.exp(band["es"])
    fig.add_trace(
        go.Scatter(
            x=band["date"],
            y=var_px,
            name=f"VaR {100 * (1 - alpha):.1f}%",
            line=dict(color=AMBER, width=1, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=band["date"],
            y=es_px,
            name=f"ES {100 * (1 - alpha):.1f}%",
            line=dict(color=RED, width=1, dash="dot"),
        )
    )
    viol = band[band["violation"]]
    if not viol.empty:
        fig.add_trace(
            go.Scatter(
                x=viol["date"],
                y=viol["close"],
                mode="markers",
                name="VaR breach",
                marker=dict(color=RED, size=7, symbol="x"),
            )
        )
if price:
    fig.add_hline(
        y=price["price"], line=dict(color=GREEN, width=1, dash="dash"), annotation_text="live"
    )

fig.update_layout(height=440, hovermode="x unified", legend=dict(orientation="h", y=1.08))
st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"VaR/ES lines are the {model_name} walk-forward forecast for that day's close, "
    "plotted against the prior close (a rolling one-step-ahead cone, not a static band). "
    "Markers = realized OOS violations (`realized < VaR`)."
)

with st.expander("Where does each number come from?"):
    st.markdown(
        """
- **Live price / 24h range** -- polled from Binance's public REST API, no key needed.
- **VaR/ES cone on the chart** -- the study's frozen `make backtest` output
  (`data/results/backtests.parquet`), one walk-forward forecast per day, refit
  daily with a 500-day rolling window.
- **"Implied range" ticker** -- the selected model re-fit *right now* on the
  latest cached window, for a same-session forecast of tomorrow's close. Falls
  back to the last frozen row if the live re-fit isn't supported for that model.
- **Model picker default** -- the FZ0-best, in-MCS model for this (asset, alpha)
  from the last `make evaluate` run (`eval_fz0_mcs.csv`).
        """
    )
