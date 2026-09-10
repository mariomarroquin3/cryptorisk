"""Phase 7 (extension): portfolio VaR / ES with a copula tail (V2_PLAN §7).

Fixed-weight BTC + ETH basket. Walk-forward, frozen OOS, same battery as the
single-asset study.

Models
------
* ``Copula-<family>``  -- GARCH(1,1)-t marginals + FHS residual inversion +
  a 2-D copula (independence / gaussian / student_t / clayton) on the joint
  tail, Monte-Carlo aggregated.
* ``Direct-<model>``   -- the univariate engine run straight on the basket
  log-return series (any model in ``registry``).

Marginals are refit every ``config.portfolio.refit_every`` days (1 by default,
to match the ``Direct-*`` models); the copula and the Monte-Carlo run daily.

Outputs:
  data/results/portfolio_backtests.parquet   tidy (date, model, alpha, var, es, ...)
  data/results/portfolio_eval.csv            coverage + ES + FZ0/MCS per model
  data/results/portfolio_subperiods.csv      the same inside each stress window
  docs/portfolio.md                          rendered (versioned)

    python -m cryptorisk.study.run_portfolio      # == make portfolio
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptorisk.backtest.engine import walk_forward
from cryptorisk.backtest.pit import berkowitz
from cryptorisk.config import load_config, repo_root
from cryptorisk.models.registry import all_models
from cryptorisk.portfolio.copula_var import CopulaVaR
from cryptorisk.portfolio.marginal import fit_marginal
from cryptorisk.study.run_evaluation import evaluate_coverage, evaluate_es, evaluate_fz0_mcs
from cryptorisk.study.subperiods import evaluate_subperiods

_PORT = "PORTFOLIO"


def _default_portfolio_cfg(cfg: dict) -> dict:
    p = cfg.get("portfolio") or {}
    p.setdefault("assets", list(cfg["assets"]))
    p.setdefault("weights", {a: 1.0 / len(p["assets"]) for a in p["assets"]})
    p.setdefault("copulas", ["independence", "gaussian", "student_t", "clayton"])
    p.setdefault("direct_models", ["FHS", "GARCH-t", "GJR-GARCH-t", "HS"])
    p.setdefault("refit_every", 1)
    p.setdefault("n_sim", 20_000)
    p.setdefault("t_df", 5.0)
    return p


def _returns(cfg: dict, assets: list[str] | None = None) -> pd.DataFrame:
    import duckdb

    assets = assets or list(cfg["assets"])
    con = duckdb.connect(str(repo_root() / cfg["paths"]["store"]), read_only=True)
    df = con.execute(
        "SELECT asset, date, log_return FROM returns_daily ORDER BY date"
    ).df()
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    wide = df.pivot(index="date", columns="asset", values="log_return")
    missing = [a for a in assets if a not in wide.columns]
    if missing:
        raise SystemExit(
            f"[portfolio] returns_daily has no rows for {missing}; "
            f"ingest them first (e.g. run_ingest --assets {' '.join(missing)} --skip-intraday)"
        )
    return wide[assets].dropna()


# --------------------------------------------------------------------------- #
def _copula_walk_forward(
    wide: pd.DataFrame, pcfg: dict, cfg: dict
) -> pd.DataFrame:
    assets = pcfg["assets"]
    alphas = cfg["alphas"]
    w = int(cfg["walk_forward"]["windows"][0])
    oos = pd.Timestamp(cfg["sample"]["oos_start"])
    refit = int(pcfg["refit_every"])
    weights = pcfg["weights"]

    models = [
        CopulaVaR(fam, assets, weights, n_sim=pcfg["n_sim"], t_df=pcfg["t_df"], seed=cfg["seed"])
        for fam in pcfg["copulas"]
    ]
    dates = wide.index.to_numpy()
    R = {a: wide[a].to_numpy(float) for a in assets}
    start = max(w, int(np.searchsorted(dates, np.datetime64(oos))))

    rows: list[dict] = []
    cache = None
    for t in range(start, len(dates)):
        win = {a: R[a][t - w : t] for a in assets}
        if cache is None or (t - start) % refit == 0:
            cache = {a: fit_marginal(win[a]) for a in assets}
        r_port = float(sum(weights[a] * R[a][t] for a in assets) / sum(weights.values()))
        for m in models:
            fc = m.fit_predict(win, alphas, marginals=cache)
            pit = float(np.mean(fc.sim <= r_port))
            for al in alphas:
                v = fc.var[al]
                rows.append({
                    "date": pd.Timestamp(dates[t]), "asset": _PORT, "model": m.name,
                    "window": w, "alpha": al, "var": v, "es": fc.es[al],
                    "sigma2": fc.sigma2, "realized": r_port,
                    "violation": bool(r_port < v), "pit": pit,
                })
    return pd.DataFrame(rows)


def _direct_walk_forward(wide: pd.DataFrame, pcfg: dict, cfg: dict) -> pd.DataFrame:
    weights = pcfg["weights"]
    tot = sum(weights.values())
    port = sum(weights[a] * wide[a] for a in pcfg["assets"]) / tot
    df = pd.DataFrame({"date": wide.index, "log_return": port.to_numpy(float)})
    wanted = set(pcfg["direct_models"])
    frames = []
    for model in (m for m in all_models() if m.name in wanted):
        res = walk_forward(
            df, model, alphas=cfg["alphas"], asset=_PORT,
            window=int(cfg["walk_forward"]["windows"][0]),
            oos_start=cfg["sample"]["oos_start"],
        )
        f = res.frame.copy()
        f["model"] = f"Direct-{model.name}"
        frames.append(f)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------- #
def _evaluate(bt: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    lvl = cfg["evaluation"]["test_level"]
    dq = cfg["evaluation"]["dq_lags"]
    mcs = cfg["evaluation"]["mcs"]
    seed = cfg["seed"]

    cov = evaluate_coverage(bt, [_PORT], lvl, dq)
    es = evaluate_es(bt, [_PORT])
    fz0 = evaluate_fz0_mcs(bt, [_PORT], mcs, seed)

    key = ["asset", "window", "alpha", "model"]
    ev = fz0.merge(
        cov[[*key, "kupiec_p", "chr_cc_p", "dq_p", "hit_rate", "passes_all", "basel_zone"]],
        on=key, how="left",
    ).merge(es[[*key, "z1", "z2", "z2_p_approx", "es_reject_approx"]], on=key, how="left")

    # Berkowitz per model (the PIT is alpha-independent; use the lower-alpha rows)
    amin = bt["alpha"].min()
    bk = []
    for model, g in bt[bt.alpha == amin].groupby("model", observed=True):
        b = berkowitz(g["pit"].to_numpy(float))
        bk.append({
            "model": model,
            "berkowitz_p": b.p_value,
            "berkowitz_reject": bool(np.isfinite(b.p_value) and b.p_value < 0.05),
        })
    ev = ev.merge(pd.DataFrame(bk), on="model", how="left")

    sub = evaluate_subperiods(bt, [_PORT], cfg)
    return ev, sub


def _summary_md(ev: pd.DataFrame, sub: pd.DataFrame, pcfg: dict, cfg: dict) -> str:
    w = pcfg["weights"]
    k = len(pcfg["assets"])
    o = [f"# Portfolio VaR / ES with a copula tail ({k}-asset basket, Phase 7)", ""]
    o.append("_`python -m cryptorisk.study.run_portfolio`._\n")
    re = pcfg["refit_every"]
    re_txt = "daily" if re == 1 else f"every {re} days"
    o.append(
        f"Fixed-weight basket: {', '.join(f'{key} {v:g}' for key, v in w.items())} "
        f"(renormalised). GARCH(1,1)-t marginals refit {re_txt}, {pcfg['n_sim']:,} "
        f"Monte-Carlo draws, {k}-dimensional copula (Student-t df {pcfg['t_df']:g}). "
        f"Same evaluation battery as the single-asset study; the joint out-of-sample "
        f"window is bounded by the shortest series (SOL, Binance spot from 2020-08).\n"
    )
    for alpha, g in ev.groupby("alpha", observed=True):
        g = g.sort_values("fz0_rank")
        o.append(f"## {100 * (1 - alpha):.1f}% VaR\n")
        o.append("| rank | model | hit rate | coverage | ES Z2 | ES ok | Berkowitz p | "
                 "mean FZ0 | in MCS | MCS p |")
        o.append("|--:|:--|--:|:--:|--:|:--:|--:|--:|:--:|--:|")
        for _, r in g.iterrows():
            o.append(
                f"| {int(r['fz0_rank'])} | {r['model']}{' (best)' if r['is_best'] else ''} | "
                f"{r['hit_rate']:.3f} | {'pass' if r['passes_all'] else 'FAIL'} | "
                f"{r['z2']:+.3f} | {'no' if r['es_reject_approx'] else 'yes'} | "
                f"{r['berkowitz_p']:.3f} | {r['fz0_mean']:.4f} | "
                f"{'yes' if r['in_mcs'] else 'no'} | {r['mcs_p']:.3f} |"
            )
        o.append("")
    if not sub.empty:
        o.append("## MCS membership by sub-period (lower alpha)\n")
        s = sub[sub.alpha == sub.alpha.min()]
        periods = list(dict.fromkeys(s["period"]))
        o.append("| model | " + " | ".join(periods) + " |")
        o.append("|:--|" + "|".join([":--:"] * len(periods)) + "|")
        base = s[s.period == "full_oos"].sort_values("fz0_rank")["model"]
        for m in base:
            marks = []
            for p in periods:
                rr = s[(s.period == p) & (s.model == m)]
                marks.append("+" if (not rr.empty and bool(rr["in_mcs"].iloc[0])) else ".")
            o.append(f"| {m} | " + " | ".join(marks) + " |")
        o.append("")
    o.append("## Read\n")
    o.extend(_read_bullets(ev, pcfg))
    o.append(
        "- The copula marginals are GARCH(1,1)-t; the residual inversion is FHS "
        "(empirical). Only the *dependence* is parametric.\n"
        "- The sub-period MCS has little power (40&ndash;90 day windows), so the "
        "copula-family differences show up mainly in the full-sample FZ0 mean.\n"
    )
    return "\n".join(o)


def _read_bullets(ev: pd.DataFrame, pcfg: dict) -> list[str]:
    """Narrative driven by the actual numbers, so the doc stays honest for any
    basket size / family set."""
    assets = pcfg["assets"]
    pair = " and ".join(assets) if len(assets) == 2 else f"the {len(assets)} assets"
    out: list[str] = []
    amin = ev["alpha"].min()
    g = ev[ev["alpha"] == amin]

    ind = g[g["model"] == "Copula-independence"]
    if not ind.empty:
        r = ind.iloc[0]
        verdict = "out of the MCS" if not bool(r["in_mcs"]) else "in the MCS but bottom-ranked"
        out.append(
            f"- **Ignoring tail dependence.** `Copula-independence` sits at hit "
            f"rate {r['hit_rate']:.3f} (target {amin:.3f}), ES Z2 {r['z2']:+.2f}, "
            f"and is {verdict} &mdash; treating {pair} as independent understates "
            f"basket tail risk.\n"
        )

    cop = g[g["model"].str.startswith("Copula-") & (g["model"] != "Copula-independence")]
    if not cop.empty:
        order = cop.sort_values("fz0_mean")["model"].str.replace("Copula-", "", regex=False).tolist()
        best = cop.sort_values("fz0_mean").iloc[0]
        out.append(
            f"- **Dependence structure.** Among the parametric copulas the FZ0 "
            f"ordering (best first) is {' < '.join(order)}; `{best['model']}` "
            f"leads with Z2 {best['z2']:+.2f}.\n"
        )

    direct = g[g["model"].str.startswith("Direct-")]
    if not direct.empty and not cop.empty:
        best_direct = direct.sort_values("fz0_mean").iloc[0]
        best_cop = cop.sort_values("fz0_mean").iloc[0]
        overall = g.sort_values("fz0_rank").iloc[0]
        if best_direct["fz0_mean"] < best_cop["fz0_mean"]:
            out.append(
                f"- **Direct vs copula.** `{best_direct['model']}` on the basket "
                f"return series still edges every copula on FZ0 "
                f"({best_direct['fz0_mean']:.4f} vs {best_cop['fz0_mean']:.4f}), "
                f"both refitting volatility daily. Overall FZ0 winner: "
                f"`{overall['model']}`.\n"
            )
        else:
            out.append(
                f"- **Direct vs copula.** The copula wins here: `{best_cop['model']}` "
                f"beats the best `Direct-*` on FZ0 ({best_cop['fz0_mean']:.4f} vs "
                f"{best_direct['fz0_mean']:.4f}). Overall FZ0 winner: "
                f"`{overall['model']}`.\n"
            )
    return out


def main() -> None:
    cfg = load_config()
    pcfg = _default_portfolio_cfg(cfg)
    res = repo_root() / cfg["paths"]["results"]
    wide = _returns(cfg, pcfg["assets"])

    print(f"[portfolio] basket = {', '.join(pcfg['assets'])} | "
          f"{len(wide)} joint days {wide.index.min().date()} -> {wide.index.max().date()}", flush=True)
    print(f"[portfolio] copula walk-forward ({', '.join(pcfg['copulas'])}) ...", flush=True)
    cop = _copula_walk_forward(wide, pcfg, cfg)
    print(f"[portfolio] direct models ({', '.join(pcfg['direct_models'])}) ...", flush=True)
    direct = _direct_walk_forward(wide, pcfg, cfg)
    bt = pd.concat([cop, direct], ignore_index=True)
    bt.to_parquet(res / "portfolio_backtests.parquet", index=False)

    print("[portfolio] evaluation ...", flush=True)
    ev, sub = _evaluate(bt, cfg)
    ev.to_csv(res / "portfolio_eval.csv", index=False)
    sub.to_csv(res / "portfolio_subperiods.csv", index=False)
    doc = repo_root() / "docs" / "portfolio.md"
    doc.write_text(_summary_md(ev, sub, pcfg, cfg), encoding="utf-8")

    print(f"[portfolio] wrote {res}/portfolio_{{backtests.parquet,eval.csv,subperiods.csv}} "
          f"and {doc}")
    show = ["alpha", "fz0_rank", "model", "hit_rate", "z2", "in_mcs"]
    print(ev.sort_values(["alpha", "fz0_rank"])[show].to_string(index=False))


if __name__ == "__main__":
    main()
