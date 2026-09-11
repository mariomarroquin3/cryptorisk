"""Phase 5 orchestration: the decision layer (V2_PLAN §6).

Turns the Phase-3 model comparison into management numbers, for the models
still in the 90% Model Confidence Set (from ``eval_fz0_mcs.csv``):

* ``decision_capital.csv`` -- FRTB ES-IMA capital per model
  (``m_c * N * |ES_97.5%,1d| * sqrt(LH)``), the square-root-of-time vs
  block-bootstrap 10-day ES, and the model-risk add-on (capital spread across
  the MCS).
* ``decision_limits.csv``  -- position limit ``N*`` from the 1-day 99% ES and a
  backtest of the limit framework (bind rate, budget breaches, ES exceedances).
* ``decision_pla.csv``     -- FRTB PLA test (RTPL vs HPL) per model + zone.
* ``decision_hedge.csv``   -- perp hedge ratios, annualised funding carry
  (+ = the short-perp hedge earns it), ES reduction (perp return proxied by
  spot -- no perp price in the store).
* ``decision_summary.md``  -- the tables rendered.

    python -m cryptorisk.study.run_decision
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptorisk.backtest.coverage import basel_traffic_light
from cryptorisk.config import load_config, repo_root
from cryptorisk.decision import capital as cap
from cryptorisk.decision import estimation_risk as er
from cryptorisk.decision import hedge as hg
from cryptorisk.decision import limits as lim
from cryptorisk.decision import pnl_attribution as pla

_A_CAPITAL = 0.025  # 97.5% ES for capital (FRTB)
_A_LIMIT = 0.01  # 99% ES for the position limit


def _mcs_members(res_dir) -> dict[tuple[str, float], list[str]]:
    fp = res_dir / "eval_fz0_mcs.csv"
    if not fp.exists():
        return {}
    f = pd.read_csv(fp)
    out: dict[tuple[str, float], list[str]] = {}
    for (asset, alpha), g in f.groupby(["asset", "alpha"]):
        out[(asset, float(alpha))] = g.loc[g["in_mcs"], "model"].tolist()
    return out


def _load(cfg) -> tuple[pd.DataFrame, dict, dict]:
    import duckdb

    res_dir = repo_root() / cfg["paths"]["results"]
    bt = pd.read_parquet(res_dir / "backtests.parquet")
    bt["date"] = pd.to_datetime(bt["date"])
    con = duckdb.connect(str(repo_root() / cfg["paths"]["store"]), read_only=True)
    ret = con.execute("SELECT asset, date, log_return FROM returns_daily ORDER BY date").df()
    fund = con.execute(
        "SELECT asset, date, funding_8h FROM microstructure_daily ORDER BY date"
    ).df()
    con.close()
    for d in (ret, fund):
        d["date"] = pd.to_datetime(d["date"])
    oos = pd.Timestamp(cfg["sample"]["oos_start"])
    ret_oos = {a: ret[(ret.asset == a) & (ret.date >= oos)] for a in cfg["assets"]}
    fund_oos = {a: fund[(fund.asset == a) & (fund.date >= oos)] for a in cfg["assets"]}
    return bt, ret_oos, fund_oos


# --------------------------------------------------------------------------- #
def capital_table(bt, mcs, cfg) -> pd.DataFrame:
    d = cfg["decision"]
    N, LH, base = d["notional_usd"], d["liquidity_horizon_days"], d["basel_multiplier_base"]
    rows = []
    for asset in cfg["assets"]:
        # capital amount uses the 97.5% ES (FRTB); the traffic-light multiplier
        # add-on is a 99% backtesting concept, so its exception count comes from
        # the alpha = 0.01 violation series.
        sl = bt[(bt.asset == asset) & (bt.window == 500) & (bt.alpha == _A_CAPITAL)]
        sl99 = bt[(bt.asset == asset) & (bt.window == 500) & (bt.alpha == _A_LIMIT)]
        viol99 = {
            m: g.sort_values("date")["violation"].to_numpy(bool) for m, g in sl99.groupby("model")
        }
        members = mcs.get((asset, _A_CAPITAL), sorted(sl.model.unique()))
        caps: dict[str, float] = {}
        for model, g in sl.groupby("model"):
            g = g.sort_values("date")
            es1d = float(g["es"].mean())
            exc = basel_traffic_light(viol99[model]).exceptions
            m_c = cap.basel_multiplier(exc, base=base)
            c = cap.es_capital(es1d, liquidity_horizon=LH, multiplier=m_c, notional=N)
            caps[model] = c
            rows.append(
                {
                    "asset": asset,
                    "model": model,
                    "in_mcs": model in members,
                    "es_975_1d": es1d,
                    "es_10d_sqrt": cap.es_horizon_sqrt_time(es1d, LH),
                    "exceptions_250d": exc,
                    "m_c": m_c,
                    "capital_usd": c,
                }
            )
        boot = cap.es_horizon_bootstrap(
            sl.drop_duplicates("date")["realized"].to_numpy(float),
            _A_CAPITAL,
            LH,
            seed=cfg["seed"],
        )
        addon = cap.model_risk_addon(caps, members)
        for r in rows:
            if r["asset"] == asset:
                r["es_10d_bootstrap"] = boot
                r["model_risk_addon_usd"] = addon
    return pd.DataFrame(rows)


def limits_table(bt, mcs, cfg) -> pd.DataFrame:
    budget = cfg["decision"]["risk_budget_es99_1d_usd"]
    rows = []
    for asset in cfg["assets"]:
        sl = bt[(bt.asset == asset) & (bt.window == 500) & (bt.alpha == _A_LIMIT)]
        members = mcs.get((asset, _A_LIMIT), sorted(sl.model.unique()))
        for model, g in sl.groupby("model"):
            g = g.sort_values("date")
            es1d = float(g["es"].mean())
            n_star = lim.max_notional(es1d, budget)
            bkt = lim.backtest_framework(
                g["es"].to_numpy(float), g["realized"].to_numpy(float), n_star, budget=budget
            )
            rows.append(
                {
                    "asset": asset,
                    "model": model,
                    "in_mcs": model in members,
                    "es_99_1d": es1d,
                    "n_star_usd": n_star,
                    "bind_rate": bkt.bind_rate,
                    "mean_utilisation": bkt.mean_utilisation,
                    "max_utilisation": bkt.max_utilisation,
                    "budget_breach_rate": bkt.budget_breach_rate,
                    "flagged_breach_rate": bkt.flagged_breach_rate,
                    "es_exceedance_rate": bkt.es_exceedance_rate,
                    "worst_loss_usd": bkt.worst_loss,
                }
            )
    return pd.DataFrame(rows)


def pla_table(bt, mcs, cfg) -> pd.DataFrame:
    N = cfg["decision"]["notional_usd"]
    rows = []
    for asset in cfg["assets"]:
        sl = bt[(bt.asset == asset) & (bt.window == 500) & (bt.alpha == _A_CAPITAL)]
        members = mcs.get((asset, _A_CAPITAL), sorted(sl.model.unique()))
        for model, g in sl.groupby("model"):
            g = g.sort_values("date")
            if g["pit"].notna().sum() < 50 or g["sigma2"].notna().sum() < 50:
                continue
            rtpl = pla.implied_rtpl(g["sigma2"].to_numpy(float), g["pit"].to_numpy(float), N)
            hpl = N * g["realized"].to_numpy(float)
            res = pla.pla_test(rtpl, hpl)
            # a real diffusion/jump split needs the model's stored jump
            # parameters (not in the parquet); leave jump_var_share at 0.
            attr = pla.attribute_pnl(
                g["realized"].to_numpy(float),
                np.zeros(len(g)),
                g["sigma2"].to_numpy(float),
                N,
            )
            rows.append(
                {
                    "asset": asset,
                    "model": model,
                    "in_mcs": model in members,
                    "pla_spearman": res.spearman,
                    "pla_ks": res.ks,
                    "pla_zone": res.zone,
                    "attr_diffusion_usd": attr.diffusion,
                    "attr_jump_usd": attr.jump,
                    "attr_residual_usd": attr.residual,
                }
            )
    return pd.DataFrame(rows)


def estimation_risk_table(cfg) -> pd.DataFrame:
    """Parameter / sampling uncertainty on the *final* estimation window, for
    three archetypes (HS, GARCH-t, FHS). The capital delta uses the Basel base
    multiplier so the three are comparable -- it isolates the estimation-risk
    contribution, not the absolute capital."""
    import duckdb

    d = cfg["decision"]
    N, LH, base = d["notional_usd"], d["liquidity_horizon_days"], d["basel_multiplier_base"]
    w = int(cfg["walk_forward"]["windows"][0])
    a_cap = _A_CAPITAL
    seed = cfg["seed"]

    con = duckdb.connect(str(repo_root() / cfg["paths"]["store"]), read_only=True)
    rows = []
    for asset in cfg["assets"]:
        r = con.execute(
            "SELECT log_return FROM returns_daily WHERE asset = ? ORDER BY date", [asset]
        ).df()["log_return"].to_numpy(float)
        r = r[np.isfinite(r)][-w:]
        if r.size < w:
            continue
        bands = {
            "HS": er.hs_band(r, [a_cap], seed=seed)[0],
            "GARCH-t": er.garch_t_band(r, [a_cap], seed=seed)[0],
            "FHS": er.fhs_band(r, [a_cap], seed=seed)[0],
        }
        for name, b in bands.items():
            rows.append(
                {
                    "asset": asset,
                    "estimator": name,
                    "alpha": a_cap,
                    "n_draws": b.n_draws,
                    "var_point": b.var_point,
                    "es_point": b.es_point,
                    "es_se": b.es_se,
                    "es_lo_p5": b.es_lo,
                    "es_hi_p95": b.es_hi,
                    "es_prudent_p5": b.es_prudent,
                    "es_widening_frac": b.es_widening(),
                    "capital_point_usd": cap.es_capital(
                        b.es_point, liquidity_horizon=LH, multiplier=base, notional=N
                    ),
                    "estimation_risk_addon_usd": er.capital_addon(
                        b, liquidity_horizon=LH, multiplier=base, notional=N
                    ),
                }
            )
    con.close()
    return pd.DataFrame(rows)


def hedge_table(ret_oos, fund_oos, cfg) -> pd.DataFrame:
    N = cfg["decision"]["notional_usd"]
    rows = []
    for asset in cfg["assets"]:
        s = ret_oos[asset][["date", "log_return"]].rename(columns={"log_return": "spot"})
        f = fund_oos[asset][["date", "funding_8h"]]
        m = s.merge(f, on="date", how="left")
        hs = hg.hedge_summary(
            m["spot"].to_numpy(float),
            m["spot"].to_numpy(float),
            m["funding_8h"].to_numpy(float),
            alpha=_A_CAPITAL,
            notional=N,
            perp_is_proxy=True,
        )
        rows.append(
            {
                "asset": asset,
                "ratio_min_var": hs.ratio_min_var,
                "ratio_es_min": hs.ratio_es_min,
                "es_unhedged": hs.es_unhedged,
                "es_hedged": hs.es_hedged,
                "es_reduction": hs.es_reduction,
                "funding_carry_annual_frac": hs.funding_carry_annual_frac,
                "funding_carry_annual_usd": hs.funding_carry_annual_usd,
                "note": hs.note,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def _summary_md(capdf, limdf, pladf, hgdf, erdf, cfg) -> str:
    d = cfg["decision"]
    o = ["# Decision layer", ""]
    o.append("_`python -m cryptorisk.study.run_decision`._\n")
    o.append(
        f"Notional ${d['notional_usd']:,.0f} | liquidity horizon {d['liquidity_horizon_days']}d "
        f"| Basel base multiplier {d['basel_multiplier_base']} "
        f"| 1-day 99% ES budget ${d['risk_budget_es99_1d_usd']:,.0f}.\n"
    )

    o.append("## Capital (FRTB ES-IMA, 97.5% ES, MCS models)\n")
    o.append("| asset | model | m_c | ES 10d √t | ES 10d boot | capital $ |")
    o.append("|:--|:--|--:|--:|--:|--:|")
    for _, r in capdf[capdf.in_mcs].sort_values(["asset", "capital_usd"]).iterrows():
        o.append(
            f"| {r['asset']} | {r['model']} | {r['m_c']:.2f} | {r['es_10d_sqrt']:.4f} | "
            f"{r['es_10d_bootstrap']:.4f} | {r['capital_usd']:,.0f} |"
        )
    for asset in cfg["assets"]:
        a = capdf[(capdf.asset == asset) & capdf.in_mcs]
        if not a.empty:
            o.append(
                f"\n{asset}: model-risk add-on (capital spread across the MCS) = "
                f"${a['model_risk_addon_usd'].iloc[0]:,.0f}.\n"
            )

    if erdf is not None and not erdf.empty:
        o.append("## Estimation-risk band (final estimation window)\n")
        o.append(
            "Parameter / sampling uncertainty on the last "
            f"{cfg['walk_forward']['windows'][0]}-day window, three archetypes: HS "
            "(stationary block bootstrap), GARCH-t (draw from the fitted "
            "asymptotic covariance, re-forecast), FHS (parameter draw + residual "
            "resample). `ES 97.5% p5` is the prudent (conservative) draw; the "
            "add-on is `capital(prudent ES) - capital(point ES)` at the Basel "
            "base multiplier, so it isolates estimation risk.\n"
        )
        o.append(
            "| asset | estimator | ES 97.5% point | ES s.e. | ES 97.5% p5 (prudent) | "
            "abs-ES widening | est.-risk add-on $ |"
        )
        o.append("|:--|:--|--:|--:|--:|--:|--:|")
        for _, r in erdf.sort_values(["asset", "estimator"]).iterrows():
            o.append(
                f"| {r['asset']} | {r['estimator']} | {r['es_point']:.4f} | "
                f"{r['es_se']:.4f} | {r['es_prudent_p5']:.4f} | "
                f"{r['es_widening_frac']:+.4f} | {r['estimation_risk_addon_usd']:,.0f} |"
            )
        o.append("")

    o.append("## Position limit N* (mean 1-day 99% ES = budget) + framework backtest\n")
    o.append(
        "`mean util` is 1.00 by construction (N* set so the *average* ES = budget); "
        "`max util` and the breach rates are the informative columns.\n"
    )
    o.append(
        "| asset | model | N* $ | bind rate | max util | budget breach | ES exceed (~1%) | worst loss $ |"
    )
    o.append("|:--|:--|--:|--:|--:|--:|--:|--:|")
    for _, r in limdf[limdf.in_mcs].sort_values(["asset", "n_star_usd"]).iterrows():
        o.append(
            f"| {r['asset']} | {r['model']} | {r['n_star_usd']:,.0f} | {r['bind_rate']:.3f} | "
            f"{r['max_utilisation']:.2f} | {r['budget_breach_rate']:.4f} | "
            f"{r['es_exceedance_rate']:.4f} | {r['worst_loss_usd']:,.0f} |"
        )
    o.append("")

    o.append("## FRTB PLA test (RTPL vs HPL)\n")
    o.append(
        "RTPL is the realized outcome mapped through the model's predictive CDF, "
        "so Spearman is ~1 by construction; the KS distance (predictive *shape* "
        "vs realized) is what separates the models.\n"
    )
    o.append("| asset | model | Spearman | KS | zone |")
    o.append("|:--|:--|--:|--:|:--:|")
    for _, r in pladf[pladf.in_mcs].sort_values(["asset", "pla_ks"]).iterrows():
        o.append(
            f"| {r['asset']} | {r['model']} | {r['pla_spearman']:.3f} | {r['pla_ks']:.3f} | "
            f"{r['pla_zone']} |"
        )
    o.append("")

    o.append("## Perpetual hedge (perp return proxied by spot)\n")
    o.append(
        "`funding carry` is the annualised funding on the hedged notional; "
        "**positive = the short-perp hedge earns it** (longs pay shorts).\n"
    )
    def _f(x, spec):
        return format(x, spec) if np.isfinite(x) else "n/a"

    o.append(
        "| asset | h (min-var) | h (ES-min) | ES unhedged | ES hedged | ES reduction | "
        "funding carry $/yr |"
    )
    o.append("|:--|--:|--:|--:|--:|--:|--:|")
    for _, r in hgdf.iterrows():
        o.append(
            f"| {r['asset']} | {_f(r['ratio_min_var'], '.3f')} | {_f(r['ratio_es_min'], '.3f')} | "
            f"{_f(r['es_unhedged'], '.4f')} | {_f(r['es_hedged'], '.4f')} | "
            f"{_f(r['es_reduction'], '.2%')} | {_f(r['funding_carry_annual_usd'], '+,.0f')} |"
        )
    o.append(f"\n_{hgdf['note'].iloc[0]}._\n")
    return "\n".join(o)


def main() -> None:
    cfg = load_config()
    res_dir = repo_root() / cfg["paths"]["results"]
    bt, ret_oos, fund_oos = _load(cfg)
    mcs = _mcs_members(res_dir)
    if not mcs:
        print(
            "[decision] eval_fz0_mcs.csv not found -- run `make evaluate` first; "
            "using all models as 'in MCS'.",
            flush=True,
        )

    print("[decision] capital ...", flush=True)
    capdf = capital_table(bt, mcs, cfg)
    capdf.to_csv(res_dir / "decision_capital.csv", index=False)

    print("[decision] limits ...", flush=True)
    limdf = limits_table(bt, mcs, cfg)
    limdf.to_csv(res_dir / "decision_limits.csv", index=False)

    print("[decision] PLA ...", flush=True)
    pladf = pla_table(bt, mcs, cfg)
    pladf.to_csv(res_dir / "decision_pla.csv", index=False)

    print("[decision] hedge ...", flush=True)
    hgdf = hedge_table(ret_oos, fund_oos, cfg)
    hgdf.to_csv(res_dir / "decision_hedge.csv", index=False)

    print("[decision] estimation risk ...", flush=True)
    erdf = estimation_risk_table(cfg)
    erdf.to_csv(res_dir / "decision_estimation_risk.csv", index=False)

    (res_dir / "decision_summary.md").write_text(
        _summary_md(capdf, limdf, pladf, hgdf, erdf, cfg), "utf-8"
    )
    print(f"[decision] wrote decision_*.csv + decision_summary.md -> {res_dir}")
    print(capdf[capdf.in_mcs][["asset", "model", "m_c", "capital_usd"]].to_string(index=False))


if __name__ == "__main__":
    main()
