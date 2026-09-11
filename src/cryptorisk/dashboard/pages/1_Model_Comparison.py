"""Model Comparison -- FZ0/MCS ranking, coverage, ES tests, one asset+alpha."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cryptorisk.config import load_config
from cryptorisk.dashboard import data as D
from cryptorisk.dashboard.theme import GREEN, MUTED, page_setup

page_setup("Model Comparison")
cfg = load_config()
st.title("Model Comparison")
st.caption(
    "FZ0 loss ranking + Model Confidence Set (90%), coverage tests, Acerbi-Szekely ES tests."
)

c1, c2 = st.columns(2)
asset = c1.selectbox("Asset", cfg["assets"])
alpha = c2.selectbox("Confidence", cfg["alphas"], format_func=lambda a: f"{100 * (1 - a):.1f}%")

results = D.load_results()
fz0 = results["fz0_mcs"]
cov = results["coverage"]
es = results["es"]

sub = fz0[(fz0.asset == asset) & (fz0.alpha == alpha)].sort_values("fz0_rank")
if sub.empty:
    st.warning("No FZ0/MCS results for this selection.")
else:
    colors = [GREEN if v else MUTED for v in sub["in_mcs"]]
    fig = go.Figure(
        go.Bar(
            x=sub["model"],
            y=sub["fz0_mean"],
            marker_color=colors,
            text=sub["fz0_rank"],
            textposition="outside",
        )
    )
    fig.update_layout(
        height=380,
        yaxis_title="Mean FZ0 loss (lower = better)",
        title=f"{asset} @ {100 * (1 - alpha):.1f}% -- green = in the 90% MCS",
    )
    st.plotly_chart(fig, use_container_width=True)

    show = sub[
        ["fz0_rank", "model", "fz0_mean", "in_mcs", "mcs_p", "dm_vs_best_p", "n_degenerate"]
    ].rename(
        columns={
            "fz0_rank": "rank",
            "fz0_mean": "FZ0",
            "in_mcs": "in MCS",
            "mcs_p": "MCS p",
            "dm_vs_best_p": "DM p (vs best)",
            "n_degenerate": "degenerate days",
        }
    )
    st.dataframe(show, use_container_width=True, hide_index=True)

st.divider()
left, right = st.columns(2)

with left:
    st.subheader("Coverage tests")
    c = cov[(cov.asset == asset) & (cov.alpha == alpha)].sort_values("model")
    if not c.empty:
        show = c[
            ["model", "hit_rate", "kupiec_p", "chr_cc_p", "dq_p", "basel_zone", "passes_all"]
        ].rename(
            columns={
                "hit_rate": "hit rate",
                "kupiec_p": "Kupiec p",
                "chr_cc_p": "Christoffersen CC p",
                "dq_p": "DQ p",
                "basel_zone": "Basel zone",
                "passes_all": "passes all",
            }
        )
        st.dataframe(
            show.style.background_gradient(
                subset=["Kupiec p", "Christoffersen CC p", "DQ p"], cmap="RdYlGn", vmin=0, vmax=0.2
            ),
            use_container_width=True,
            hide_index=True,
        )
    st.caption(f"Nominal miss rate at this alpha: {alpha:.1%}. p < 0.05 rejects correct coverage.")

with right:
    st.subheader("Expected Shortfall tests (Acerbi-Szekely)")
    e = es[(es.asset == asset) & (es.alpha == alpha)].sort_values("model")
    if not e.empty:
        show = e[
            ["model", "n_breach", "z1", "z1_p_approx", "z2", "z2_p_approx", "es_reject_approx"]
        ].rename(
            columns={
                "n_breach": "breaches",
                "z1_p_approx": "Z1 p",
                "z2_p_approx": "Z2 p",
                "es_reject_approx": "ES rejected",
            }
        )
        st.dataframe(
            show.style.background_gradient(
                subset=["Z1 p", "Z2 p"], cmap="RdYlGn", vmin=0, vmax=0.2
            ),
            use_container_width=True,
            hide_index=True,
        )
    st.caption(
        "Asymptotic-normal p-values (no per-day predictive draws available for a simulated null)."
    )

st.divider()
st.subheader("Giacomini-White conditional predictive ability (regime-conditioned)")
gw = results["gw_cpa"]
g = (
    gw[(gw.asset == asset) & (gw.alpha == alpha)]
    if not gw.empty and "alpha" in gw.columns
    else gw[gw.asset == asset]
    if not gw.empty
    else pd.DataFrame()
)
if not g.empty:
    st.dataframe(g, use_container_width=True, hide_index=True)
else:
    st.caption("No GW-CPA output for this selection.")
