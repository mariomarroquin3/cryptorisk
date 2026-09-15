"""Phase 6: assemble the empirical results report (V2_PLAN §8).

Deterministic. Reads the Phase 3-5 artefacts in ``data/results/`` plus the
store, and writes

* ``docs/results.md``            -- the results write-up (a supervisor's read)
* ``docs/model_cards/<slug>.md`` -- one card per model (spec pointer + OOS scorecard)
* ``docs/figures/*.png``         -- the figures embedded in the above (gitignored)

``docs/methodology.tex`` carries the mathematics; this report carries the
numbers. Nothing here is random: it only tabulates committed pipeline output.

    python -m cryptorisk.study.report      # == make report
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cryptorisk.config import load_config, repo_root  # noqa: E402

_FIGSIZE = (9, 5)
plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})

# one-line idea + model-specific limitations, keyed by registry name.
_MODEL_NOTES: dict[str, tuple[str, list[str]]] = {
    "HS": (
        "Empirical quantile of the last w returns; no distributional assumption.",
        [
            "Reacts slowly to a regime change (the whole window must roll over).",
            "The tail is only as resolved as `alpha*w` observations.",
        ],
    ),
    "AWHS": (
        "Historical Simulation with exponentially decaying weights (half-life 125d).",
        [
            "Effective sample size ~ H/ln2 << w, so the tail estimate is noisier.",
            "Adapts to the recency of the tail, not its shape.",
        ],
    ),
    "EWMA": (
        "RiskMetrics: one-parameter recursive variance, Gaussian (or fixed-nu t) tail.",
        [
            "Integrated (alpha+beta=1): no mean reversion, no finite unconditional variance.",
            "The Gaussian tail is too thin at alpha=0.01 -- see the ES test.",
        ],
    ),
    "GARCH-t": (
        "GARCH(1,1) variance recursion, standardized Student-t innovations.",
        [
            "Symmetric response to good and bad news.",
            "One regime: a single (alpha, beta) for calm and crisis alike.",
        ],
    ),
    "GJR-GARCH-t": (
        "GARCH-t plus a leverage term (bad news raises variance more).",
        ["The asymmetry is a single extra parameter, constant over time."],
    ),
    "EGARCH-t": (
        "Models log-variance, so positivity is free and the sign asymmetry is direct.",
        [
            "The log-variance recursion can collapse toward 0 (guarded by a vol floor) "
            "or blow up near the non-stationary boundary (vol ceiling).",
            "Absorbs excess kurtosis into the recursion, leaving nu large and the "
            "1% quantile thin.",
        ],
    ),
    "FHS": (
        "GARCH filter, then the empirical tail of the standardized residuals.",
        [
            "Inherits any misspecification of the GARCH volatility dynamics.",
            "The residual tail is still a finite sample -- sparse at alpha=0.01.",
        ],
    ),
    "GARCH-EVT": (
        "GARCH filter, then a Generalized Pareto fit to the residual tail (McNeil-Frey).",
        [
            "Sensitive to the threshold (fixed here at the 90th loss percentile).",
            "The GPD shape parameter is estimated from few exceedances.",
        ],
    ),
    "Jump-Diffusion": (
        "Brownian part plus a compound-Poisson jump component (Merton).",
        [
            "Jump intensity and size are constant -- no clustering of jumps.",
            "VaR/ES come from a fixed-seed Monte-Carlo sample, so a small "
            "quantisation error remains.",
        ],
    ),
    "HAR-RV": (
        "OLS of realized variance on its daily / weekly / monthly averages (Corsi).",
        [
            "Targets the conditional *mean* of RV; RV is right-skewed, so the model "
            "runs thin for a lower return quantile.",
            "Linear -- no volatility-of-volatility term.",
        ],
    ),
    "HARQ": (
        "HAR-RV with the daily lag shrunk on days when RV was measured noisily.",
        [
            "Best variance forecaster, near-worst for VaR/ES: the tail is a point "
            "forecast with a thin t on top.",
            "The RQ interaction is z-scored using window-wide moments (mild leak).",
        ],
    ),
    "Realized-GARCH": (
        "GARCH driven by observed RV, with a measurement equation tying "
        "the two (Hansen-Huang-Shek).",
        [
            "Eight parameters -- more to estimate than plain GARCH.",
            "Assumes a Gaussian measurement error for log RV.",
        ],
    ),
    "Realized-SV": (
        "Heston-style CIR variance filtered with an Unscented Kalman Filter "
        "from the realized measure alone (log RV_t = zeta + phi*log V_t + "
        "u_t, phi fitted not fixed at 1), plus a GJR-style lagged leverage "
        "term. The return does not update the filtered state -- it scores "
        "its own exact Student-t log-density given the predicted V instead "
        "of going through the approximate Harvey-Ruiz-Shephard log-square "
        "trick tried during development, which stayed biased even after "
        "correcting its Gaussian constants for fat tails. VaR/ES come from "
        "a GPD tail (McNeil-Frey, 2000, same construction as GARCH-EVT) "
        "fitted to the filtered window's own standardized residuals, not "
        "the fitted Student-t directly -- a single nu from the whole-sample "
        "likelihood undersold the most extreme moves.",
        [
            "Leverage is lagged (gamma * I(r_{t-1}<0) * resid_{t-1}^2 in the "
            "CIR drift), not Heston's same-day correlated shocks (rho != 0) "
            "-- avoids a same-day causality problem in the filter, at the "
            "cost of a one-day-slower asymmetric response.",
            "Quasi-MLE via an approximate filter likelihood (Gaussian on "
            "the RV channel only), not the exact CIR transition density.",
        ],
    ),
    "GARCH-X": (
        "GARCH-t with lagged RV added to the variance equation.",
        [
            "Degrades to plain GARCH-t (gamma=0) when the realized block is absent.",
            "The exogenous term is a single contemporaneous lag.",
        ],
    ),
    "CAViaR-SAV": (
        "Autoregression of the quantile itself (symmetric absolute value), fit by the tick loss.",
        [
            "No full density: no PIT / Berkowitz, and ES is a scaled quantile.",
            "Symmetric news impact -- unlike AS it cannot react differently to good vs bad days.",
            "Fit separately per alpha; cross-alpha monotonicity is repaired ex-post.",
        ],
    ),
    "CAViaR-AS": (
        "Autoregression of the quantile itself (asymmetric slope), fit by the tick loss.",
        [
            "No full density: no PIT / Berkowitz, and ES is a scaled quantile.",
            "Fit separately per alpha; cross-alpha monotonicity is repaired ex-post.",
        ],
    ),
    "CAViaR-X-AS": (
        "CAViaR-AS plus sqrt(RV_{t-1}) as an exogenous quantile driver.",
        [
            "Same density / ES caveats as CAViaR-AS.",
            "Adds one parameter estimated by a derivative-free search.",
        ],
    ),
    "MS-GARCH": (
        "Two-regime Markov-switching GARCH mixture, walk-forward in R.",
        [
            "The walk-forward regime probability has ~no out-of-sample signal "
            "(see the regime-identification section); the layer is descriptive.",
            "Served from a cache; dates without a prediction fall back to empirical.",
        ],
    ),
}

_METHOD_FAMILY = {
    "HS": "Non-parametric",
    "AWHS": "Non-parametric",
    "EWMA": "Exponential smoothing",
    "GARCH-t": "Single-regime GARCH",
    "GJR-GARCH-t": "Single-regime GARCH",
    "EGARCH-t": "Single-regime GARCH",
    "FHS": "Semiparametric tail",
    "GARCH-EVT": "Semiparametric tail",
    "Jump-Diffusion": "Discontinuous",
    "HAR-RV": "Realized-measure",
    "HARQ": "Realized-measure",
    "Realized-GARCH": "Realized-measure",
    "Realized-SV": "Stochastic volatility",
    "GARCH-X": "Exogenous / conditional",
    "CAViaR-SAV": "Exogenous / conditional",
    "CAViaR-AS": "Exogenous / conditional",
    "CAViaR-X-AS": "Exogenous / conditional",
    "MS-GARCH": "Regime-switching",
}


# --------------------------------------------------------------------------- #
def _slug(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def _load() -> dict:
    cfg = load_config()
    res = repo_root() / cfg["paths"]["results"]
    w0 = cfg["walk_forward"]["windows"][0]  # the report covers the primary window only

    def csv(n: str) -> pd.DataFrame:
        p = res / n
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_csv(p)
        # eval CSVs carry a `window` column; the report tabulates one window, so
        # groupby(asset, alpha) below is unambiguous.
        return df[df["window"] == w0] if "window" in df.columns else df

    import duckdb

    con = duckdb.connect(str(repo_root() / cfg["paths"]["store"]), read_only=True)
    ret = con.execute("SELECT asset, date, log_return FROM returns_daily ORDER BY date").df()
    rv = con.execute("SELECT asset, date, rv FROM realized_daily ORDER BY date").df()
    con.close()
    for d in (ret, rv):
        d["date"] = pd.to_datetime(d["date"])

    bt = pd.read_parquet(res / "backtests.parquet")
    bt = bt[bt["window"] == w0].copy()
    bt["date"] = pd.to_datetime(bt["date"])

    return {
        "cfg": cfg,
        "res": res,
        "ret": ret,
        "rv": rv,
        "bt": bt,
        "coverage": csv("eval_coverage.csv"),
        "es": csv("eval_es.csv"),
        "fz0": csv("eval_fz0_mcs.csv"),
        "density": csv("eval_density.csv"),
        "vol": csv("eval_volforecast.csv"),
        "sub": csv("eval_subperiods.csv"),
        "gw": csv("eval_gw_cpa.csv"),
        "regime": csv("regime_identification.csv"),
        "cap": csv("decision_capital.csv"),
        "lim": csv("decision_limits.csv"),
        "pla": csv("decision_pla.csv"),
        "hedge": csv("decision_hedge.csv"),
        "er": csv("decision_estimation_risk.csv"),
    }


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _figdir():
    d = repo_root() / "docs" / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def fig_returns_rv(D: dict) -> str:
    oos = pd.Timestamp(D["cfg"]["sample"]["oos_start"])
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, asset in zip(np.atleast_1d(axes), D["cfg"]["assets"], strict=False):
        r = D["ret"][D["ret"].asset == asset]
        v = D["rv"][D["rv"].asset == asset].set_index("date")["rv"].rolling(21).mean()
        ax.plot(r["date"], r["log_return"], lw=0.4, color="0.5")
        ax2 = ax.twinx()
        ax2.plot(v.index, np.sqrt(v) * np.sqrt(365), lw=1.0, color="C3")
        ax2.grid(False)
        ax.axvline(oos, color="C0", ls="--", lw=1)
        ax.set_title(
            f"{asset}: log-return (grey) and 21d annualised realized vol (red); "
            f"OOS from the dashed line"
        )
        ax.set_ylabel("log-return")
        ax2.set_ylabel("ann. vol")
    fig.tight_layout()
    p = _figdir() / "fig01_returns_rv.png"
    fig.savefig(p)
    plt.close(fig)
    return p.name


def fig_var_path(D: dict, asset: str = "BTC", alpha: float = 0.025) -> str:
    bt = D["bt"]
    keep = ["Realized-GARCH", "HS", "EWMA", "EGARCH-t"]
    sl = bt[(bt.asset == asset) & (bt.window == 500) & (bt.alpha == alpha)]
    if sl.empty:
        return ""
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    real = sl.drop_duplicates("date").set_index("date")["realized"]
    ax.plot(real.index, real, lw=0.4, color="0.6", label="realized")
    for m, c in zip(keep, ["C0", "C1", "C2", "C3"], strict=False):
        g = sl[sl.model == m].set_index("date")["var"]
        if not g.empty:
            ax.plot(g.index, g, lw=0.9, color=c, label=f"VaR {m}")
    ax.set_title(f"{asset}: {100 * (1 - alpha):.1f}% 1-day VaR vs realized return")
    ax.set_ylabel("log-return")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    p = _figdir() / "fig02_var_path.png"
    fig.savefig(p)
    plt.close(fig)
    return p.name


def fig_fz0_mcs(D: dict) -> str:
    fz0 = D["fz0"]
    if fz0.empty:
        return ""
    cells = list(fz0.groupby(["asset", "alpha"], observed=True))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, ((asset, alpha), g) in zip(axes.ravel(), cells, strict=False):
        g = g.sort_values("fz0_mean")
        colors = ["C0" if m else "0.7" for m in g["in_mcs"]]
        ax.barh(g["model"], g["fz0_mean"], color=colors)
        ax.invert_yaxis()
        ax.set_title(f"{asset}  |  {100 * (1 - alpha):.1f}% VaR   (blue = in 90% MCS)")
        ax.set_xlabel("mean FZ0 loss (lower is better)")
    fig.suptitle("Headline: FZ0 ranking and Model Confidence Set", y=1.01)
    fig.tight_layout()
    p = _figdir() / "fig03_fz0_mcs.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p.name


def fig_qlike(D: dict) -> str:
    vol = D["vol"]
    if vol.empty:
        return ""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, asset in zip(np.atleast_1d(axes), D["cfg"]["assets"], strict=False):
        g = vol[vol.asset == asset].sort_values("qlike")
        colors = ["C2" if m else "0.7" for m in g["in_mcs_qlike"]]
        ax.barh(g["model"], g["qlike"], color=colors)
        ax.invert_yaxis()
        ax.set_title(f"{asset}: QLIKE vs RV (green = in QLIKE-MCS)")
        ax.set_xlabel("mean QLIKE (lower is better)")
    fig.tight_layout()
    p = _figdir() / "fig04_qlike.png"
    fig.savefig(p)
    plt.close(fig)
    return p.name


def fig_regime(D: dict) -> str:
    reg = D["regime"]
    if reg.empty:
        return ""
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    piv = reg.pivot_table(index="series", columns="asset", values="corr_absret")
    piv = piv.reindex([s for s in ["filt_wf", "pred_wf", "insample"] if s in piv.index])
    piv.plot.bar(ax=ax)
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_title("MS-GARCH: corr(P(high-vol regime), |return|)")
    ax.set_ylabel("Pearson correlation")
    ax.set_xticklabels(
        [
            {
                "filt_wf": "walk-forward\n(filtered)",
                "pred_wf": "walk-forward\n(predicted)",
                "insample": "full-sample fit\n(in-sample)",
            }[s]
            for s in piv.index
        ],
        rotation=0,
    )
    fig.tight_layout()
    p = _figdir() / "fig05_regime_identification.png"
    fig.savefig(p)
    plt.close(fig)
    return p.name


def fig_capital(D: dict) -> str:
    cap = D["cap"]
    if cap.empty:
        return ""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, asset in zip(np.atleast_1d(axes), D["cfg"]["assets"], strict=False):
        g = cap[(cap.asset == asset) & cap.in_mcs].sort_values("capital_usd")
        ax.barh(g["model"], g["capital_usd"] / 1e3, color="C4")
        ax.invert_yaxis()
        addon = g["model_risk_addon_usd"].iloc[0] / 1e3 if not g.empty else 0.0
        ax.set_title(
            f"{asset}: FRTB ES-IMA capital (MCS models)\nmodel-risk add-on = ${addon:,.0f}k"
        )
        ax.set_xlabel("capital ($k)")
    fig.tight_layout()
    p = _figdir() / "fig06_capital.png"
    fig.savefig(p)
    plt.close(fig)
    return p.name


def fig_pit(D: dict) -> str:
    bt, dens = D["bt"], D["density"]
    if dens.empty:
        return ""
    b = dens[dens.asset == "BTC"] if (dens.asset == "BTC").any() else dens
    best = b.sort_values("berkowitz_p", ascending=False).iloc[0]["model"]
    worst = b.sort_values("berkowitz_p").iloc[0]["model"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), [best, worst], strict=False):
        u = bt[(bt.asset == "BTC") & (bt.window == 500) & (bt.alpha == 0.025) & (bt.model == m)][
            "pit"
        ].dropna()
        ax.hist(u, bins=20, range=(0, 1), color="C0", edgecolor="w")
        ax.axhline(len(u) / 20, color="C3", ls="--", lw=1)
        row = b[b.model == m].iloc[0]
        ax.set_title(f"{m}  (Berkowitz p = {row['berkowitz_p']:.3f})")
        ax.set_xlabel("PIT")
    axes[0].set_ylabel("count")
    fig.suptitle("BTC PIT histograms: best- vs worst-calibrated density (uniform = flat)")
    fig.tight_layout()
    p = _figdir() / "fig07_pit.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p.name


def all_figures(D: dict) -> dict[str, str]:
    return {
        "returns_rv": fig_returns_rv(D),
        "var_path": fig_var_path(D),
        "fz0_mcs": fig_fz0_mcs(D),
        "qlike": fig_qlike(D),
        "regime": fig_regime(D),
        "capital": fig_capital(D),
        "pit": fig_pit(D),
    }


# --------------------------------------------------------------------------- #
# results.md
# --------------------------------------------------------------------------- #
def _mcs_line(g: pd.DataFrame) -> str:
    inside = g[g["in_mcs"]].sort_values("fz0_rank")["model"].tolist()
    best = g[g["is_best"]]["model"].iloc[0]
    return f"**{best}** best; MCS ({len(inside)}/{len(g)}): {', '.join(inside)}"


def _fig(figs: dict[str, str], key: str, caption: str) -> str:
    return f"![{caption}](figures/{figs[key]})\n" if figs.get(key) else ""


def results_md(D: dict, figs: dict[str, str]) -> str:
    cfg = D["cfg"]
    A = cfg["assets"]
    oos = cfg["sample"]["oos_start"]
    fz0 = D["fz0"]
    o: list[str] = []
    P = o.append

    P("# cryptorisk: empirical results\n")
    P(
        "_Generated by `python -m cryptorisk.study.report` from the committed "
        "pipeline output. The mathematics is in [`methodology.tex`](methodology.tex); "
        "this document is the numbers._\n"
    )

    P("## Executive summary\n")
    if not fz0.empty:
        for (asset, alpha), g in fz0.groupby(["asset", "alpha"], observed=True):
            P(f"- **{asset} {100 * (1 - alpha):.1f}% VaR** &mdash; {_mcs_line(g)}.")
        P("")
        P(
            f"- The 90% Model Confidence Set is **wide** in every cell: over "
            f"~{int(fz0['n'].max()):,} out-of-sample days the test cannot separate "
            f"the middle of the field."
        )
    es = D["es"]
    if not es.empty:
        bad = sorted(es[es["es_reject_approx"]]["model"].unique())
        P(
            f"- **Expected Shortfall**: {', '.join(bad) or 'no model'} fail the "
            f"Acerbi&ndash;Sz&eacute;kely Z2 test (ES too optimistic); the GARCH "
            f"family and the realized-measure models pass."
        )
    P(
        "- **Volatility forecasting** and **tail forecasting** disagree: HARQ / "
        "HAR-RV lead on QLIKE but sit near the bottom on FZ0 &mdash; a good "
        "variance point forecast is not a good tail."
    )
    reg = D["regime"]
    if not reg.empty:
        wf = reg[reg.series == "filt_wf"].set_index("asset")["corr_absret"]
        ins = reg[reg.series == "insample"].set_index("asset")["corr_absret"]
        P(
            f"- **MS-GARCH regime identification**: corr(P(high-vol), |r|) is "
            f"{wf.get('BTC', float('nan')):.2f} walk-forward vs "
            f"{ins.get('BTC', float('nan')):.2f} full-sample for BTC "
            f"({wf.get('ETH', float('nan')):.2f} vs {ins.get('ETH', float('nan')):.2f} "
            f"for ETH). The real-time regime signal has no content; the layer is "
            f"descriptive only."
        )
    cap = D["cap"]
    if not cap.empty:
        add = cap.groupby("asset")["model_risk_addon_usd"].first()
        P(
            f"- **Decision layer**: the model-risk add-on (capital spread across "
            f"the MCS) is ${add.get('BTC', 0) / 1e3:,.0f}k (BTC) / "
            f"${add.get('ETH', 0) / 1e3:,.0f}k (ETH) on a "
            f"${cfg['decision']['notional_usd'] / 1e6:.0f}M notional."
        )
    P("")

    P("## Data and protocol\n")
    P(
        f"BTC and ETH, daily {cfg['sample']['start']} onward (CoinMetrics reference "
        f"rate, reconciled against Binance), plus ~1.8M 5-minute bars for the "
        f"realized measures. Out-of-sample from **{oos}** (frozen before any result "
        f"was seen), ~{int(fz0['n'].max()) if not fz0.empty else 0:,} days per asset. "
        f"Rolling window {cfg['walk_forward']['windows'][0]} days, refit daily; "
        f"MS-GARCH refit every 20 days in R. alpha in "
        f"{{{', '.join(str(a) for a in cfg['alphas'])}}}. Seed {cfg['seed']}.\n"
    )
    P(_fig(figs, "returns_rv", "returns and realized vol"))

    cov = D["coverage"]
    P("## 1. VaR coverage\n")
    P(
        "Kupiec unconditional coverage, Christoffersen conditional coverage, and "
        "the Engle&ndash;Manganelli Dynamic Quantile test, at the 5% level. "
        "`passes all` = none of the three rejects.\n"
    )
    for (asset, alpha), g in cov.groupby(["asset", "alpha"], observed=True):
        g = g.sort_values("model")
        P(f"### {asset} &mdash; {100 * (1 - alpha):.1f}% VaR\n")
        P("| model | hit rate | Kupiec p | Chr. cc p | DQ p | Basel | passes all |")
        P("|:--|--:|--:|--:|--:|:--:|:--:|")
        for _, r in g.iterrows():
            bz = r["basel_zone"] if isinstance(r["basel_zone"], str) and r["basel_zone"] else "-"
            P(
                f"| {r['model']} | {r['hit_rate']:.3f} | {r['kupiec_p']:.3f} | "
                f"{r['chr_cc_p']:.3f} | {r['dq_p']:.3f} | {bz} | "
                f"{'yes' if r['passes_all'] else 'NO'} |"
            )
        passers = g[g["passes_all"]]["model"].tolist()
        P(f"\nPass every coverage test: {', '.join(passers) if passers else 'none'}.\n")
    P(_fig(figs, "var_path", "VaR paths"))

    P("## 2. Expected Shortfall\n")
    P(
        "Acerbi&ndash;Sz&eacute;kely Z1 (conditional on a breach) and Z2 (joint "
        "frequency + magnitude). Z2 < 0 means realized tail losses are worse than "
        "the ES forecast. p-values are the asymptotic-normal approximation.\n"
    )
    for (asset, alpha), g in D["es"].groupby(["asset", "alpha"], observed=True):
        g = g.sort_values("z2")
        P(f"### {asset} &mdash; {100 * (1 - alpha):.1f}%\n")
        P("| model | breaches | Z1 | Z2 | Z2 p | ES optimistic |")
        P("|:--|--:|--:|--:|--:|:--:|")
        for _, r in g.iterrows():
            P(
                f"| {r['model']} | {int(r['n_breach'])} | {r['z1']:.3f} | {r['z2']:+.3f} | "
                f"{r['z2_p_approx']:.3f} | {'yes' if r['es_reject_approx'] else ''} |"
            )
        P("")

    P("## 3. Headline: FZ0 ranking and the Model Confidence Set\n")
    P(
        "The Fissler&ndash;Ziegel FZ0 loss is strictly consistent for the "
        "(VaR, ES) pair, so its out-of-sample mean is a legitimate ranking. The "
        "90% MCS (Hansen&ndash;Lunde&ndash;Nason, stationary block bootstrap) is "
        "the set that cannot be separated from the best. `DM vs best p` is the "
        "Diebold&ndash;Mariano two-sided p-value against the winner.\n"
    )
    for (asset, alpha), g in fz0.groupby(["asset", "alpha"], observed=True):
        g = g.sort_values("fz0_rank")
        P(
            f"### {asset} &mdash; {100 * (1 - alpha):.1f}%   "
            f"(dropped {int(g['n_degenerate'].iloc[0])} degenerate days)\n"
        )
        P("| rank | model | mean FZ0 | in MCS | MCS p | DM vs best p |")
        P("|--:|:--|--:|:--:|--:|--:|")
        for _, r in g.iterrows():
            star = " (best)" if r["is_best"] else ""
            dm = "-" if not np.isfinite(r["dm_vs_best_p"]) else f"{r['dm_vs_best_p']:.3f}"
            P(
                f"| {int(r['fz0_rank'])} | {r['model']}{star} | {r['fz0_mean']:.4f} | "
                f"{'yes' if r['in_mcs'] else 'no'} | {r['mcs_p']:.3f} | {dm} |"
            )
        P("")
    P(_fig(figs, "fz0_mcs", "FZ0 and MCS"))

    P("## 4. Volatility forecasting\n")
    P(
        "QLIKE of the one-step variance forecast against 5-minute realized "
        "variance, its own 90% MCS, and the Mincer&ndash;Zarnowitz regression "
        "`RV = a + b*sigma2` with the joint test of `(a, b) = (0, 1)`.\n"
    )
    for asset, g in D["vol"].groupby("asset", observed=True):
        g = g.sort_values("qlike")
        P(f"### {asset}\n")
        P("| rank | model | QLIKE | in MCS | MZ b | MZ R2 | MZ (a,b)=(0,1) p |")
        P("|--:|:--|--:|:--:|--:|--:|--:|")
        for _, r in g.iterrows():
            P(
                f"| {int(r['qlike_rank'])} | {r['model']} | {r['qlike']:.4f} | "
                f"{'yes' if r['in_mcs_qlike'] else 'no'} | {r['mz_b']:.2f} | "
                f"{r['mz_r2']:.3f} | {r['mz_joint_p']:.3f} |"
            )
        P("")
    P(_fig(figs, "qlike", "QLIKE"))

    P("## 5. Density calibration (Berkowitz)\n")
    P(
        "On the PIT `u_t = F_t(r_t)`: transform to `z_t = Phi^-1(u_t)`, fit an "
        "AR(1), LR-test `(mu, rho, sigma2) = (0, 0, 1)` ~ chi-square(3). CAViaR "
        "and MS-GARCH have no density and are omitted.\n"
    )
    for asset, g in D["density"].groupby("asset", observed=True):
        g = g.sort_values("berkowitz_p", ascending=False)
        P(f"### {asset}\n")
        P("| model | Berkowitz LR | p | rho | sigma2 | reject |")
        P("|:--|--:|--:|--:|--:|:--:|")
        for _, r in g.iterrows():
            P(
                f"| {r['model']} | {r['berkowitz_lr']:.2f} | {r['berkowitz_p']:.3f} | "
                f"{r['rho']:+.3f} | {r['sigma2']:.3f} | {'yes' if r['reject'] else ''} |"
            )
        P("")
    P(_fig(figs, "pit", "PIT histograms"))

    sub = D["sub"]
    P("## 6. Sub-periods and conditional predictive ability\n")
    if not sub.empty:
        P(
            "FZ0 + MCS re-run inside each ex-ante window. The windows are short "
            "(40&ndash;92 trading days), so an all-in MCS is expected &mdash; and "
            "that is the finding: the ranking does not detectably change across "
            "regimes.\n"
        )
        for (asset, alpha), g in sub[sub.alpha == sub.alpha.min()].groupby(
            ["asset", "alpha"], observed=True
        ):
            periods = list(dict.fromkeys(g["period"]))
            P(f"### {asset} &mdash; {100 * (1 - alpha):.1f}%   MCS membership by period\n")
            P("| model | " + " | ".join(periods) + " |")
            P("|:--|" + "|".join([":--:"] * len(periods)) + "|")
            base = g[g.period == "full_oos"].sort_values("fz0_rank")["model"]
            for m in base:
                marks = []
                for p in periods:
                    rr = g[(g.period == p) & (g.model == m)]
                    marks.append("+" if (not rr.empty and bool(rr["in_mcs"].iloc[0])) else ".")
                P(f"| {m} | " + " | ".join(marks) + " |")
            nd = g.groupby("period")["n_days"].first().to_dict()
            P("\n" + ", ".join(f"{p}: {nd[p]}d" for p in periods) + ".\n")
    gw = D["gw"]
    if not gw.empty:
        P(
            "Giacomini&ndash;White test of equal *conditional* predictive ability "
            "(FZ0 loss, instrument `[1, z(log RV_{t-1})]`), plus a HAC t-test of "
            "the loss differential's slope on the RV state.\n"
        )
        P("| asset | a | best vs challenger | mean gap | GW p | slope t | slope p | edge vs RV |")
        P("|:--|--:|:--|--:|--:|--:|--:|:--|")
        for _, r in gw.iterrows():
            P(
                f"| {r['asset']} | {r['alpha']:g} | {r['model_a']} vs {r['model_b']} | "
                f"{r['mean_fz0_gap']:+.4f} | {r['gw_p']:.3f} | {r['rvz_t']:+.2f} | "
                f"{r['rvz_p']:.3f} | {r['best_edge_vs_rv']} |"
            )
        P("")

    P("## 7. MS-GARCH regime identification\n")
    if not reg.empty:
        P(
            "Correlation of each regime-probability series with the realized "
            "volatility state. `filt_wf` / `pred_wf` are the walk-forward W=500 "
            "filtered / predicted probabilities; `insample` is a single "
            "full-sample fit.\n"
        )
        P("| asset | series | corr(\\|r\\|) | Spearman(\\|r\\|) | corr(RV) | corr(RV21) |")
        P("|:--|:--|--:|--:|--:|--:|")
        for _, r in reg.iterrows():
            P(
                f"| {r['asset']} | {r['series']} | {r['corr_absret']:.3f} | "
                f"{r['spearman_absret']:.3f} | {r['corr_rv']:.3f} | {r['corr_rv21']:.3f} |"
            )
        P(
            "\nThe full-sample fit tracks |r| strongly (Spearman ~0.8&ndash;0.9); "
            "the walk-forward probability does not (|corr| < 0.06). The 2-regime "
            "model needs more data than a 500-day window to separate the states in "
            "real time. The window sweep (`regime-id --run-r`) is a multi-hour R "
            "job and is not run here.\n"
        )
        P(_fig(figs, "regime", "regime identification"))

    P("## 8. Decision layer\n")
    d = cfg["decision"]
    P(
        f"Notional ${d['notional_usd']:,.0f}, liquidity horizon "
        f"{d['liquidity_horizon_days']} days, Basel base multiplier "
        f"{d['basel_multiplier_base']}, 1-day 99% ES budget "
        f"${d['risk_budget_es99_1d_usd']:,.0f}. Tables are the MCS models.\n"
    )
    if not cap.empty:
        P("### Capital (FRTB ES-IMA, 97.5% ES)\n")
        P(
            "The capital amount uses the 97.5% ES; the multiplier `m_c` (Basel "
            "base + traffic-light add-on) is a 99% concept, so its exception "
            "count comes from the alpha=0.01 backtest.\n"
        )
        P("| asset | model | m_c | ES 10d (sqrt-t) | ES 10d (bootstrap) | capital $ |")
        P("|:--|:--|--:|--:|--:|--:|")
        for _, r in cap[cap.in_mcs].sort_values(["asset", "capital_usd"]).iterrows():
            P(
                f"| {r['asset']} | {r['model']} | {r['m_c']:.2f} | {r['es_10d_sqrt']:.4f} | "
                f"{r['es_10d_bootstrap']:.4f} | {r['capital_usd']:,.0f} |"
            )
        for a in A:
            gg = cap[(cap.asset == a) & cap.in_mcs]
            if not gg.empty:
                P(f"\n{a}: model-risk add-on = ${gg['model_risk_addon_usd'].iloc[0]:,.0f}.")
        P("")
        P(_fig(figs, "capital", "capital"))

    erd = D.get("er", pd.DataFrame())
    if not erd.empty:
        P("### Estimation risk (final estimation window)\n")
        P(
            "Parameter / sampling uncertainty on the last "
            f"{cfg['walk_forward']['windows'][0]}-day window, for three archetypes. "
            "HS is a stationary block bootstrap of the window; GARCH-t draws the "
            "parameters from the fitted asymptotic covariance and re-forecasts "
            "(no refit); FHS combines a parameter draw for the vol path with a "
            "residual resample. `ES p5` is the prudent (5th-percentile) draw; the "
            "add-on is `capital(prudent ES) &minus; capital(point ES)` at the "
            "Basel base multiplier, isolating the estimation-risk contribution.\n"
        )
        P("| asset | estimator | ES 97.5% point | ES s.e. | ES p5 (prudent) | est.-risk add-on $ |")
        P("|:--|:--|--:|--:|--:|--:|")
        for _, r in erd.sort_values(["asset", "estimator"]).iterrows():
            P(
                f"| {r['asset']} | {r['estimator']} | {r['es_point']:.4f} | "
                f"{r['es_se']:.4f} | {r['es_prudent_p5']:.4f} | "
                f"{r['estimation_risk_addon_usd']:,.0f} |"
            )
        add = erd["estimation_risk_addon_usd"].dropna()
        mr = cap.groupby("asset")["model_risk_addon_usd"].first() if not cap.empty else None
        mr_txt = (
            f" &mdash; smaller than the model-risk add-on (${mr.max() / 1e3:,.0f}k)"
            if mr is not None and len(mr)
            else ""
        )
        P(
            f"\nThe estimation-risk add-on ranges ${add.min() / 1e3:,.0f}k&ndash;"
            f"${add.max() / 1e3:,.0f}k across the three archetypes and two assets"
            f"{mr_txt}. It bounds the §9 caveat: estimation risk is real but, at "
            "the decision layer, second-order.\n"
        )
    if not D["lim"].empty:
        P("### Position limit N* and the framework backtest\n")
        P(
            "`mean util` is 1.00 by construction; the breach rates and `max util` "
            "carry the signal.\n"
        )
        P("| asset | model | N* $ | max util | budget breach | ES exceed | worst loss $ |")
        P("|:--|:--|--:|--:|--:|--:|--:|")
        for _, r in D["lim"][D["lim"].in_mcs].sort_values(["asset", "n_star_usd"]).iterrows():
            P(
                f"| {r['asset']} | {r['model']} | {r['n_star_usd']:,.0f} | "
                f"{r['max_utilisation']:.2f} | {r['budget_breach_rate']:.4f} | "
                f"{r['es_exceedance_rate']:.4f} | {r['worst_loss_usd']:,.0f} |"
            )
        P("")
    if not D["pla"].empty:
        P("### FRTB PLA test\n")
        P(
            "RTPL is the realized outcome mapped through the model's predictive "
            "CDF, so Spearman is ~1 by construction; the KS distance carries the "
            "signal.\n"
        )
        P("| asset | model | Spearman | KS | zone |")
        P("|:--|:--|--:|--:|:--:|")
        for _, r in D["pla"][D["pla"].in_mcs].sort_values(["asset", "pla_ks"]).iterrows():
            P(
                f"| {r['asset']} | {r['model']} | {r['pla_spearman']:.3f} | "
                f"{r['pla_ks']:.3f} | {r['pla_zone']} |"
            )
        P("")
    if not D["hedge"].empty:
        P("### Perpetual hedge (perp return proxied by spot)\n")
        P(
            "Funding carry is annualised; **positive = the short-perp hedge earns "
            "it** (longs pay shorts). No perp price in the store, so the min-var "
            "ratio sits at 1.0 and the ES-minimising hedge / ES reduction are not "
            "meaningful against an identical series (`n/a`); basis risk is "
            "unavailable.\n"
        )

        def _hf(x, spec):
            return format(x, spec) if np.isfinite(x) else "n/a"

        P("| asset | h (min-var) | h (ES-min) | ES unhedged | ES hedged | funding carry $/yr |")
        P("|:--|--:|--:|--:|--:|--:|")
        for _, r in D["hedge"].iterrows():
            P(
                f"| {r['asset']} | {_hf(r['ratio_min_var'], '.3f')} | "
                f"{_hf(r['ratio_es_min'], '.3f')} | {_hf(r['es_unhedged'], '.4f')} | "
                f"{_hf(r['es_hedged'], '.4f')} | "
                f"{_hf(r['funding_carry_annual_usd'], '+,.0f')} |"
            )
        P("")

    P("## 9. What the numbers do not settle\n")
    P(
        "- **The MCS is wide.** With one asset-pair and ~2,700 heavy-tailed days "
        "the test cannot rank the middle of the field. A firmer answer needs more "
        "assets, a longer sample, or a rolling-origin MCS stability analysis."
    )
    P(
        "- **Multiple testing.** 17 models x 2 assets x 2 alpha, no family-wise "
        "correction; the sub-period split makes this worse (n drops fast)."
    )
    P(
        "- **Estimation risk is only bounded, not propagated.** The daily "
        "backtests use the parameter point estimate. §8 quantifies the "
        "parameter / sampling uncertainty on the final estimation window for "
        "three archetypes (a prudent-percentile capital add-on); it is not "
        "carried through every day of every model."
    )
    P("- **ES p-values are approximate** (asymptotic normal, not simulated).")
    P(
        "- **The main study is one asset at a time, spot only** &mdash; no "
        "options and the perp hedge uses spot as a price proxy. A fixed-weight "
        "4-asset basket (BTC/ETH/SOL/BNB) with a k-dimensional copula tail is "
        "the Phase-7 extension in [`portfolio.md`](portfolio.md)."
    )
    P(
        "- Model-specific caveats are in the model cards and in "
        "[`methodology.tex`](methodology.tex) §9.\n"
    )

    P("## Reproducibility\n")
    P(
        "`make data && make msgarch && make backtest && make evaluate && "
        "make subperiods && make regime-id && make decide && make report && "
        "make portfolio` rebuilds every artefact from the sources, "
        "deterministically (seed in `config/study.yaml`). `make data` also "
        "daily-ingests the portfolio basket's extra assets, so `make portfolio` "
        "needs nothing further. The store and "
        "`data/results/` are gitignored; the text of this report, the model "
        "cards and `portfolio.md` are versioned, the figures are regenerated.\n"
    )
    return "\n".join(o)


# --------------------------------------------------------------------------- #
# model cards
# --------------------------------------------------------------------------- #
def _card(name: str, D: dict) -> str:
    idea, lims = _MODEL_NOTES.get(name, ("", []))
    fam = _METHOD_FAMILY.get(name, "")
    cov, es, fz0 = D["coverage"], D["es"], D["fz0"]
    o: list[str] = []
    P = o.append
    P(f"# Model card: {name}\n")
    P(f"**Family:** {fam}  \n**Idea:** {idea}\n")

    P("## Out-of-sample scorecard\n")
    P("| asset | a | hit rate | coverage | ES Z2 | ES ok | FZ0 rank | in MCS |")
    P("|:--|--:|--:|:--:|--:|:--:|--:|:--:|")
    for asset in D["cfg"]["assets"]:
        for alpha in sorted(D["cfg"]["alphas"]):
            c = cov[(cov.asset == asset) & (cov.alpha == alpha) & (cov.model == name)]
            e = es[(es.asset == asset) & (es.alpha == alpha) & (es.model == name)]
            f = fz0[(fz0.asset == asset) & (fz0.alpha == alpha) & (fz0.model == name)]
            if c.empty or f.empty:
                continue
            c, f = c.iloc[0], f.iloc[0]
            z2 = f"{e.iloc[0]['z2']:+.2f}" if not e.empty else "-"
            esok = "no" if (not e.empty and e.iloc[0]["es_reject_approx"]) else "yes"
            n_cell = len(fz0[(fz0.asset == asset) & (fz0.alpha == alpha)])
            P(
                f"| {asset} | {alpha:g} | {c['hit_rate']:.3f} | "
                f"{'pass' if c['passes_all'] else 'FAIL'} | {z2} | {esok} | "
                f"{int(f['fz0_rank'])}/{n_cell} | {'yes' if f['in_mcs'] else 'no'} |"
            )
    P("")

    dd = D["density"][D["density"].model == name]
    if not dd.empty:
        P(
            "**Density (Berkowitz):** "
            + ", ".join(
                f"{r['asset']} p={r['berkowitz_p']:.3f}{' (reject)' if r['reject'] else ''}"
                for _, r in dd.iterrows()
            )
            + "\n"
        )
    else:
        P("**Density:** no full predictive density (quantile-only model).\n")

    vv = D["vol"][D["vol"].model == name]
    if not vv.empty:
        P(
            "**Volatility forecast (QLIKE):** "
            + ", ".join(
                f"{r['asset']} rank {int(r['qlike_rank'])}/"
                f"{len(D['vol'][D['vol'].asset == r['asset']])} (MZ b={r['mz_b']:.2f})"
                for _, r in vv.iterrows()
            )
            + "\n"
        )

    cc = D["cap"][D["cap"].model == name]
    ll = D["lim"][D["lim"].model == name]
    if not cc.empty and not ll.empty:
        P(
            "**Decision layer:** "
            + "; ".join(
                f"{r['asset']} capital ${r['capital_usd']:,.0f} (m_c {r['m_c']:.2f}), "
                f"N* ${ll[ll.asset == r['asset']]['n_star_usd'].iloc[0]:,.0f}"
                for _, r in cc.iterrows()
            )
            + "\n"
        )

    if lims:
        P("## Known limitations\n")
        for x in lims:
            P(f"- {x}")
        P("")
    P(
        "_The estimator, likelihood and closed-form VaR/ES are in "
        "[`../methodology.tex`](../methodology.tex)._\n"
    )
    return "\n".join(o)


def model_cards(D: dict) -> list[str]:
    from cryptorisk.models.registry import all_models

    d = repo_root() / "docs" / "model_cards"
    d.mkdir(parents=True, exist_ok=True)
    written = []
    for m in all_models():
        p = d / f"{_slug(m.name)}.md"
        p.write_text(_card(m.name, D), encoding="utf-8")
        written.append(p.name)
    (d / "README.md").write_text(
        "# Model cards\n\nOne card per model in `registry.all_models()`, "
        "auto-generated by `python -m cryptorisk.study.report`. Each card is the "
        "model's idea, its out-of-sample scorecard (coverage / ES / FZ0-MCS / "
        "density / QLIKE / decision layer) and its known limitations. The "
        "mathematics is in [`../methodology.tex`](../methodology.tex).\n",
        encoding="utf-8",
    )
    return written


# --------------------------------------------------------------------------- #
def main() -> None:
    D = _load()
    print("[report] figures ...", flush=True)
    figs = all_figures(D)
    print("[report] results.md ...", flush=True)
    (repo_root() / "docs" / "results.md").write_text(results_md(D, figs), encoding="utf-8")
    print("[report] model cards ...", flush=True)
    cards = model_cards(D)
    n_fig = sum(1 for v in figs.values() if v)
    print(
        f"[report] wrote docs/results.md, docs/model_cards/ ({len(cards)} cards), "
        f"docs/figures/ ({n_fig} figures)"
    )


if __name__ == "__main__":
    main()
