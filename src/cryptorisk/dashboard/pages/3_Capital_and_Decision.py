"""Capital & Decision -- FRTB ES-IMA capital, limits, hedge, estimation risk."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cryptorisk.config import load_config
from cryptorisk.dashboard import data as D
from cryptorisk.dashboard.theme import AMBER, RED, page_setup

page_setup("Capital & Decision")
cfg = load_config()
st.title("Capital & Decision")
st.caption("FRTB ES-IMA capital, position limits, estimation-risk add-on, perp hedge.")

results = D.load_results()
cap = results["capital"]
er = results["estimation_risk"]
lim = results["limits"]
hedge = results["hedge"]

asset = st.selectbox("Asset", cfg["assets"])

# --------------------------------------------------------------------- #
# Capital stack
# --------------------------------------------------------------------- #
c = cap[cap.asset == asset]
e = er[er.asset == asset]
if not c.empty:
    row = c[c["in_mcs"]].sort_values("capital_usd").iloc[0] if c["in_mcs"].any() else c.iloc[0]
    # `model_risk_addon_usd` is NOT a component already baked into `capital_usd`
    # -- it's a separate, asset-level scalar (max-min capital across the MCS,
    # same value on every row for that asset; see run_decision.capital_table /
    # decision.capital.model_risk_addon). The stack below is additive on top
    # of this model's own point capital, not a decomposition of it.
    point_capital = float(row["capital_usd"])
    model_addon = float(row["model_risk_addon_usd"])
    est_addon = float(e["estimation_risk_addon_usd"].median()) if not e.empty else 0.0
    total_capital = point_capital + model_addon

    fig = go.Figure(
        go.Bar(
            x=["Point ES capital", "+ model-risk add-on", "+ estimation-risk add-on (median)"],
            y=[point_capital, model_addon, est_addon],
            marker_color=["#4dabf7", AMBER, RED],
        )
    )
    fig.update_layout(
        height=340,
        yaxis_title="USD (per $1M notional)",
        title=f"{asset} capital stack -- {row['model']}",
    )
    st.plotly_chart(fig, use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model", str(row["model"]))
    m2.metric(
        "Point + model-risk capital",
        f"${total_capital:,.0f}",
        help="This model's own point ES capital plus the MCS spread add-on.",
    )
    m3.metric(
        "Model-risk add-on",
        f"${model_addon:,.0f}",
        help="capital spread (max - min) across the in-MCS models for this asset, not specific to the model shown.",
    )
    m4.metric("Basel exceptions (250d)", f"{int(row['exceptions_250d'])} ({row['m_c']}x)")

    st.dataframe(
        c[
            [
                "model",
                "in_mcs",
                "es_975_1d",
                "es_10d_sqrt",
                "es_10d_bootstrap",
                "exceptions_250d",
                "m_c",
                "capital_usd",
                "model_risk_addon_usd",
            ]
        ]
        .sort_values("capital_usd")
        .rename(
            columns={
                "in_mcs": "in MCS",
                "es_975_1d": "ES 97.5% (1d)",
                "es_10d_sqrt": "ES (10d, sqrt-t)",
                "es_10d_bootstrap": "ES (10d, bootstrap)",
                "exceptions_250d": "exceptions",
                "m_c": "Basel m_c",
                "capital_usd": "capital $",
                "model_risk_addon_usd": "model-risk add-on $",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.warning("No decision-layer output. Run `make decide`.")

st.divider()

# --------------------------------------------------------------------- #
# Estimation risk
# --------------------------------------------------------------------- #
st.subheader("Estimation-risk band (parameter-uncertainty bootstrap)")
if not e.empty:
    fig2 = go.Figure()
    fig2.add_trace(
        go.Bar(
            x=e["estimator"], y=e["capital_point_usd"], name="point capital", marker_color="#4dabf7"
        )
    )
    fig2.add_trace(
        go.Bar(
            x=e["estimator"],
            y=e["estimation_risk_addon_usd"],
            name="estimation-risk add-on",
            marker_color=RED,
        )
    )
    fig2.update_layout(barmode="stack", height=320, yaxis_title="USD")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "Prudent ES = 5th-percentile of the bootstrap draws (HS: block bootstrap; "
        "GARCH-t / FHS: parameter draw from the fitted covariance, no refit). "
        "Add-on is second-order next to the model-risk add-on above."
    )
else:
    st.caption("No estimation-risk output for this asset.")

st.divider()

# --------------------------------------------------------------------- #
# Limits and hedge
# --------------------------------------------------------------------- #
l1, l2 = st.columns(2)
with l1:
    st.subheader("Position limits")
    ls = lim[lim.asset == asset]
    if not ls.empty:
        st.dataframe(
            ls[
                [
                    "model",
                    "in_mcs",
                    "n_star_usd",
                    "bind_rate",
                    "mean_utilisation",
                    "budget_breach_rate",
                    "worst_loss_usd",
                ]
            ].rename(
                columns={
                    "in_mcs": "in MCS",
                    "n_star_usd": "limit $",
                    "bind_rate": "bind rate",
                    "mean_utilisation": "avg util.",
                    "budget_breach_rate": "budget breach rate",
                    "worst_loss_usd": "worst loss $",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
with l2:
    st.subheader("Perp hedge")
    hs = hedge[hedge.asset == asset]
    if not hs.empty:
        r = hs.iloc[0]
        st.metric("Min-variance hedge ratio", f"{r['ratio_min_var']:.3f}")
        st.metric(
            "Funding carry (annualised)",
            f"{r['funding_carry_annual_frac']:.2%}",
            f"${r['funding_carry_annual_usd']:,.0f}/yr",
        )
        if pd.notna(r.get("es_reduction")):
            st.metric("ES reduction from hedge", f"{r['es_reduction']:.1%}")
        else:
            st.caption(f"Note: {r.get('note', 'n/a')}")
