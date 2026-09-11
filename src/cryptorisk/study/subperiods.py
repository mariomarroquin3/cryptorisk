"""Sub-period re-evaluation + conditional predictive ability (V2_PLAN §5.5).

Two questions Phase 3's full-sample tables cannot answer:

1. **Does the ranking survive a regime change?**  Re-run the FZ0 + Model
   Confidence Set and the coverage / ES tests inside each *ex-ante* stress
   window (COVID 2020-03, Luna/UST 2022-05, FTX 2022-11) and a calm window
   (``calm_23``), and report whether the MCS composition changes.  Windows are
   short (40-70 trading days) so the MCS is deliberately underpowered there --
   an all-in MCS is itself the finding.

2. **Is one model conditionally better in turbulence?**  Giacomini-White (2006)
   test of equal *conditional* predictive ability on the FZ0 loss, with the
   conditioning instrument ``h_{t-1} = [1, z(log RV_{t-1})]`` (yesterday's
   realized-vol level, standardised).  A rejection means the accuracy gap
   between the two models depends on the volatility state.

    python -m cryptorisk.study.subperiods
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from cryptorisk.backtest.scoring import fz0_loss, giacomini_white
from cryptorisk.config import load_config, repo_root
from cryptorisk.study.run_evaluation import evaluate_coverage, evaluate_es, evaluate_fz0_mcs

# models probed against the per-cell FZ0 winner in the GW-CPA test
_GW_CHALLENGERS = ("MS-GARCH", "HS", "EWMA", "HARQ", "GARCH-t", "HAR-RV")


def _periods(cfg: dict) -> dict[str, tuple[str, str]]:
    out = {"full_oos": (cfg["sample"]["oos_start"], "2100-01-01")}
    out.update({k: (v[0], v[1]) for k, v in cfg["evaluation"]["subperiods"].items()})
    return out


def evaluate_subperiods(bt: pd.DataFrame, assets: list[str], cfg: dict) -> pd.DataFrame:
    mcs_cfg = dict(cfg["evaluation"]["mcs"])
    level = cfg["evaluation"]["test_level"]
    dq_lags = cfg["evaluation"]["dq_lags"]
    seed = cfg["seed"]

    frames = []
    for name, (d0, d1) in _periods(cfg).items():
        m = (bt["date"] >= pd.Timestamp(d0)) & (bt["date"] <= pd.Timestamp(d1))
        sl = bt[m]
        n_days = sl["date"].nunique()
        if n_days < 25:
            continue
        this = dict(mcs_cfg)  # short windows: shrink the bootstrap block to fit
        this["block_bootstrap_len"] = max(2, min(mcs_cfg["block_bootstrap_len"], n_days // 4))

        fz0 = evaluate_fz0_mcs(sl, assets, this, seed)
        cov = evaluate_coverage(sl, assets, level, dq_lags)
        es = evaluate_es(sl, assets)

        key = ["asset", "window", "alpha", "model"]
        merged = fz0.merge(
            cov[[*key, "kupiec_p", "chr_cc_p", "dq_p", "hit_rate", "passes_all"]],
            on=key,
            how="left",
        ).merge(es[[*key, "z2", "z2_p_approx", "es_reject_approx"]], on=key, how="left")
        merged.insert(0, "period", name)
        merged["n_days"] = n_days
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)


def _rv_instrument(db: str, asset: str, dates: pd.Series) -> np.ndarray:
    """``[1, z(log RV_{t-1})]`` aligned to ``dates``."""
    import duckdb

    con = duckdb.connect(db, read_only=True)
    rv = con.execute(
        "SELECT date, rv FROM realized_daily WHERE asset = ? ORDER BY date", [asset]
    ).df()
    con.close()
    rv["date"] = pd.to_datetime(rv["date"])
    s = rv.set_index("date")["rv"].reindex(pd.to_datetime(dates))
    lr = np.log(s.to_numpy(float))
    lr_lag = np.concatenate([[np.nan], lr[:-1]])
    z = (lr_lag - np.nanmean(lr_lag)) / np.nanstd(lr_lag)
    return np.column_stack([np.ones_like(z), np.nan_to_num(z, nan=0.0)])


def _hac_slope_test(d: np.ndarray, x: np.ndarray, lag: int) -> tuple[float, float, float]:
    """OLS ``d = b0 + b1 x``; HAC (Newey-West) t-test of ``b1 = 0``.

    ``b1`` is how the loss differential ``L_best - L_chal`` moves with the
    conditioning signal ``x`` (lagged z-scored log RV): ``b1 > 0`` means the
    best model's advantage *shrinks* when RV is high.
    """
    xm = np.column_stack([np.ones_like(x), x])
    n = x.size
    xtx_inv = np.linalg.inv(xm.T @ xm)
    beta = xtx_inv @ xm.T @ d
    u = d - xm @ beta
    s = xm * u[:, None]
    meat = s.T @ s
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1)
        gk = s[k:].T @ s[:-k]
        meat += w * (gk + gk.T)
    cov = xtx_inv @ meat @ xtx_inv
    se1 = float(np.sqrt(cov[1, 1]))
    if se1 == 0 or not np.isfinite(se1):
        return float(beta[1]), np.nan, np.nan
    t = float(beta[1] / se1)
    return float(beta[1]), t, float(2.0 * stats.t.sf(abs(t), df=n - 2))


def evaluate_gw_cpa(bt: pd.DataFrame, assets: list[str], cfg: dict) -> pd.DataFrame:
    db = str(repo_root() / cfg["paths"]["store"])
    oos = pd.Timestamp(cfg["sample"]["oos_start"])
    rows = []
    for asset in assets:
        for window in sorted(bt["window"].unique(), key=str):
            for alpha in sorted(bt["alpha"].unique()):
                sl = bt[
                    (bt["asset"] == asset)
                    & (bt["window"] == window)
                    & (bt["alpha"] == alpha)
                    & (bt["date"] >= oos)
                ]
                piv_v = sl.pivot_table(index="date", columns="model", values="var").dropna(axis=1)
                piv_e = sl.pivot_table(index="date", columns="model", values="es").dropna(axis=1)
                models = sorted(set(piv_v.columns) & set(piv_e.columns))
                idx = piv_v.index
                # drop days where ANY surviving model emitted a degenerate
                # forecast (non-negative VaR / ES not strictly negative) --
                # same rule as evaluate_fz0_mcs, so a numerical artefact can't
                # dominate the mean FZ0 gap or the GW statistic.
                ok = np.ones(len(idx), dtype=bool)
                for m in models:
                    v = piv_v.loc[idx, m].to_numpy(float)
                    e = piv_e.loc[idx, m].to_numpy(float)
                    ok &= np.isfinite(v) & np.isfinite(e) & (v < 0) & (e < -1e-6)
                idx = idx[ok]
                realized = (
                    sl.drop_duplicates("date").set_index("date")["realized"].reindex(idx)
                ).to_numpy(float)
                losses = {
                    m: fz0_loss(
                        realized, piv_v.loc[idx, m].to_numpy(float),
                        piv_e.loc[idx, m].to_numpy(float), alpha,
                    )
                    for m in models
                }
                best = min(models, key=lambda m: float(np.nanmean(losses[m])))
                h = _rv_instrument(db, asset, pd.Series(idx))
                rvz = h[:, 1]
                lag = max(1, int(round(len(idx) ** (1 / 3))))
                lvl = cfg["evaluation"]["test_level"]
                for chal in _GW_CHALLENGERS:
                    if chal == best or chal not in models:
                        continue
                    diff = losses[best] - losses[chal]
                    # joint GW test (equal conditional predictive ability); for a
                    # 1-step forecast the loss diff is an MDS under H0, so lag=0.
                    gw = giacomini_white(losses[best], losses[chal], h, hac_lag=0)
                    b1, t1, p1 = _hac_slope_test(diff, rvz, lag)
                    edge = "flat"
                    if np.isfinite(p1) and p1 < lvl:
                        edge = "shrinks in high RV" if b1 > 0 else "grows in high RV"
                    rows.append(
                        {
                            "asset": asset,
                            "window": window,
                            "alpha": alpha,
                            "model_a": best,
                            "model_b": chal,
                            "mean_fz0_gap": float(np.mean(diff)),
                            "gw_stat": gw.statistic,
                            "gw_p": gw.p_value,
                            "gw_reject": gw.rejects(lvl),
                            "rvz_slope": b1,
                            "rvz_t": t1,
                            "rvz_p": p1,
                            "best_edge_vs_rv": edge,
                        }
                    )
    return pd.DataFrame(rows)


def _summary_md(sub: pd.DataFrame, gw: pd.DataFrame) -> str:
    out = ["# Sub-period re-evaluation & conditional predictive ability", ""]
    out.append("_`python -m cryptorisk.study.subperiods`._\n")
    for (asset, window, alpha), g in sub.groupby(["asset", "window", "alpha"], observed=True):
        out.append(f"## {asset} | window {window} | alpha {alpha:g}\n")
        periods = list(dict.fromkeys(g["period"]))
        out.append("MCS membership by period (`+` in set, `.` out):\n")
        out.append("| model | " + " | ".join(periods) + " |")
        out.append("|:--|" + "|".join([":--:"] * len(periods)) + "|")
        base = g[g["period"] == "full_oos"].sort_values("fz0_rank")["model"]
        for m in base:
            marks = []
            for p in periods:
                r = g[(g["period"] == p) & (g["model"] == m)]
                marks.append("+" if (not r.empty and bool(r["in_mcs"].iloc[0])) else ".")
            out.append(f"| {m} | " + " | ".join(marks) + " |")
        nd = g.groupby("period")["n_days"].first().to_dict()
        out.append("\n" + ", ".join(f"{p}: {nd[p]}d" for p in periods) + ".\n")

    if not gw.empty:
        out.append("## Conditional predictive ability (FZ0 loss)\n")
        out.append(
            "Giacomini-White joint test of *equal conditional* predictive ability "
            "(instrument `[1, z(log RV_{t-1})]`), plus a HAC t-test of the RV-state "
            "slope of the loss differential `L_best - L_chal`. The joint test is "
            "dominated by the unconditional gap (already in DM); the slope is the "
            "regime-dependence.\n"
        )
        out.append(
            "| asset | a | best vs challenger | mean gap | GW p | slope t | slope p | best edge vs RV |"
        )
        out.append("|:--|--:|:--|--:|--:|--:|--:|:--|")
        for _, r in gw.iterrows():
            out.append(
                f"| {r['asset']} | {r['alpha']:g} | {r['model_a']} vs {r['model_b']} | "
                f"{r['mean_fz0_gap']:+.4f} | {r['gw_p']:.3f} | {r['rvz_t']:+.2f} | "
                f"{r['rvz_p']:.3f} | {r['best_edge_vs_rv']} |"
            )
        out.append("")
    return "\n".join(out)


def run() -> None:
    cfg = load_config()
    res_dir = repo_root() / cfg["paths"]["results"]
    bt = pd.read_parquet(res_dir / "backtests.parquet")
    bt["date"] = pd.to_datetime(bt["date"])
    assets = cfg["assets"]

    print("[subperiods] per-window batteries ...", flush=True)
    sub = evaluate_subperiods(bt, assets, cfg)
    sub.to_csv(res_dir / "eval_subperiods.csv", index=False)

    print("[subperiods] Giacomini-White conditional predictive ability ...", flush=True)
    gw = evaluate_gw_cpa(bt, assets, cfg)
    gw.to_csv(res_dir / "eval_gw_cpa.csv", index=False)

    (res_dir / "eval_subperiods.md").write_text(_summary_md(sub, gw), encoding="utf-8")
    print(
        f"[subperiods] wrote eval_subperiods.csv / eval_gw_cpa.csv / eval_subperiods.md -> {res_dir}"
    )

    top3 = sub[sub.period == "full_oos"].sort_values("fz0_rank")["model"].head(3).tolist()
    keep = ["asset", "alpha", "period", "model", "fz0_rank", "in_mcs", "n_days"]
    piv = sub[(sub.alpha == sub.alpha.min()) & (sub.model.isin(top3))][keep]
    print(piv.to_string(index=False))


if __name__ == "__main__":
    run()
