"""Portfolio -- 4-asset basket VaR/ES, copula comparison."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cryptorisk.config import load_config
from cryptorisk.dashboard import data as D
from cryptorisk.dashboard.theme import GREEN, MUTED, page_setup

page_setup("Portfolio")
cfg = load_config()
port_cfg = cfg.get("portfolio", {})
basket = port_cfg.get("assets", [])
weights = port_cfg.get("weights", {})

st.title("Portfolio")
st.caption(
    f"Equal-weight basket ({', '.join(f'{a} {weights.get(a, 0):.0%}' for a in basket)}), "
    "copula tail vs. independence, evaluated with the same FZ0/MCS battery."
)

results = D.load_results()
peval = results["portfolio_eval"]
alphas = cfg["alphas"]

alpha = st.selectbox("Confidence", alphas, format_func=lambda a: f"{100 * (1 - a):.1f}%")
sub = (
    peval[(peval.asset == "PORTFOLIO") & (peval.alpha == alpha)].sort_values("fz0_rank")
    if not peval.empty
    else pd.DataFrame()
)

if sub.empty:
    st.warning("No portfolio evaluation output found. Run `make portfolio`.")
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
        title=f"Basket @ {100 * (1 - alpha):.1f}%",
    )
    st.plotly_chart(fig, use_container_width=True)

    show = sub[
        [
            "fz0_rank",
            "model",
            "fz0_mean",
            "in_mcs",
            "hit_rate",
            "kupiec_p",
            "z2",
            "basel_zone",
            "passes_all",
        ]
    ].rename(
        columns={
            "fz0_rank": "rank",
            "fz0_mean": "FZ0",
            "in_mcs": "in MCS",
            "hit_rate": "hit rate",
            "kupiec_p": "Kupiec p",
            "basel_zone": "Basel zone",
            "passes_all": "passes all",
        }
    )
    st.dataframe(show, use_container_width=True, hide_index=True)

    copulas = sub[sub["model"].str.contains("Copula", case=False, na=False)]
    directs = sub[sub["model"].str.startswith("Direct-")]
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Copula family comparison")
        if not copulas.empty:
            st.dataframe(
                copulas[["model", "fz0_rank", "fz0_mean", "hit_rate", "z2"]].rename(
                    columns={"fz0_rank": "rank", "fz0_mean": "FZ0", "hit_rate": "hit rate"}
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Copula-independence ignoring tail dependence typically over-breaches badly at 99%."
            )
    with c2:
        st.subheader("Direct univariate models on the basket return")
        if not directs.empty:
            st.dataframe(
                directs[["model", "fz0_rank", "fz0_mean", "hit_rate"]].rename(
                    columns={"fz0_rank": "rank", "fz0_mean": "FZ0", "hit_rate": "hit rate"}
                ),
                use_container_width=True,
                hide_index=True,
            )

st.divider()
st.subheader("Live basket composition")
cols = st.columns(len(basket) or 1)
prices = {}
for i, a in enumerate(basket):
    p = D.live_price(a)
    prices[a] = p
    w = weights.get(a, 1 / max(len(basket), 1))
    with cols[i]:
        if p:
            st.metric(f"{a} ({w:.0%})", f"${p['price']:,.2f}", f"{p['change_pct']:+.2f}%")
        else:
            st.metric(f"{a} ({w:.0%})", "n/a")

if all(prices.values()):
    basket_chg = sum(weights.get(a, 0) * prices[a]["change_pct"] for a in basket)
    st.metric("Basket 24h change (weighted)", f"{basket_chg:+.2f}%")
