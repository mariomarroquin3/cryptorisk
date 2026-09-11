"""Phase 3 orchestration: the full evaluation battery over
``data/results/backtests.parquet`` (V2_PLAN §5).

Produces, per (asset, window, alpha) unless noted:

* ``eval_coverage.csv``    - Kupiec, Christoffersen ind/cc, Engle-Manganelli DQ,
                             Basel traffic light (alpha = 0.01).
* ``eval_es.csv``          - Acerbi-Szekely Z1/Z2 with the asymptotic one-sided
                             p-value (simulation p-value pending model draws).
* ``eval_fz0_mcs.csv``     - the headline: mean FZ0 loss, rank, Diebold-Mariano
                             vs the best model, and Model Confidence Set
                             membership + MCS p-value.
* ``eval_density.csv``     - Berkowitz LR on the PIT (models that emit a PIT).
* ``eval_volforecast.csv`` - QLIKE / MSE / QLIKE-MCS / Mincer-Zarnowitz
                             (delegated to :mod:`cryptorisk.study.vol_forecast_eval`).
* ``eval_summary.md``      - the tables above rendered for a quick read.

    python -m cryptorisk.study.run_evaluation
    python -m cryptorisk.study.run_evaluation --assets BTC --skip-volforecast
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from cryptorisk.backtest.coverage import (
    basel_traffic_light,
    christoffersen_cc,
    christoffersen_independence,
    dq_test,
    kupiec_pof,
)
from cryptorisk.backtest.es_tests import z1, z1_pvalue_asymptotic, z2, z2_pvalue_asymptotic
from cryptorisk.backtest.pit import berkowitz
from cryptorisk.backtest.scoring import diebold_mariano, fz0_loss, model_confidence_set
from cryptorisk.config import load_config, repo_root
from cryptorisk.study.vol_forecast_eval import run as run_vol_forecast


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _slice(bt: pd.DataFrame, asset: str, window, alpha: float) -> pd.DataFrame:
    s = bt[(bt["asset"] == asset) & (bt["window"] == window) & (bt["alpha"] == alpha)]
    return s.sort_values(["model", "date"])


def _panel(sub: pd.DataFrame, value: str) -> pd.DataFrame:
    """date x model wide panel of ``value``, keeping only fully-observed models."""
    w = sub.pivot_table(index="date", columns="model", values=value)
    return w.dropna(axis=1, how="any")


def _grid(bt: pd.DataFrame, assets: list[str]) -> list[tuple]:
    windows = sorted(bt["window"].unique(), key=str)
    alphas = sorted(bt["alpha"].unique())
    return [(a, w, al) for a in assets for w in windows for al in alphas]


# --------------------------------------------------------------------------- #
# batteries
# --------------------------------------------------------------------------- #
def evaluate_coverage(
    bt: pd.DataFrame, assets: list[str], level: float, dq_lags: int
) -> pd.DataFrame:
    rows = []
    for asset, window, alpha in _grid(bt, assets):
        sub = _slice(bt, asset, window, alpha)
        for model, g in sub.groupby("model", observed=True):
            # `violation` is NaN (not True/False) on a day the engine could not
            # form a finite VaR; `.to_numpy(bool)` would upcast that NaN to
            # True and silently count a non-forecast as a breach.
            viol = g["violation"].fillna(False).to_numpy(bool)
            var = g["var"].to_numpy(float)
            kp = kupiec_pof(viol, alpha)
            ci = christoffersen_independence(viol)
            cc = christoffersen_cc(viol, alpha)
            dq = dq_test(viol, var, alpha, lags=dq_lags)
            basel = basel_traffic_light(viol) if abs(alpha - 0.01) < 1e-9 else None
            rows.append(
                {
                    "asset": asset,
                    "window": window,
                    "alpha": alpha,
                    "model": model,
                    "n": int(viol.size),
                    "hit_rate": float(viol.mean()),
                    "kupiec_p": kp.p_value,
                    "kupiec_reject": kp.rejects(level),
                    "chr_ind_p": ci.p_value,
                    "chr_ind_reject": ci.rejects(level),
                    "chr_cc_p": cc.p_value,
                    "chr_cc_reject": cc.rejects(level),
                    "dq_p": dq.p_value,
                    "dq_reject": dq.rejects(level),
                    "basel_zone": basel.zone if basel else "",
                    "basel_addon": basel.multiplier_addon if basel else np.nan,
                    "passes_all": not (
                        kp.rejects(level)
                        or ci.rejects(level)
                        or cc.rejects(level)
                        or dq.rejects(level)
                    ),
                }
            )
    return pd.DataFrame(rows)


def evaluate_es(bt: pd.DataFrame, assets: list[str]) -> pd.DataFrame:
    rows = []
    for asset, window, alpha in _grid(bt, assets):
        sub = _slice(bt, asset, window, alpha)
        for model, g in sub.groupby("model", observed=True):
            r = g["realized"].to_numpy(float)
            v = g["var"].to_numpy(float)
            e = g["es"].to_numpy(float)
            z2v = z2(r, v, e, alpha)
            z2p = z2_pvalue_asymptotic(r, v, e, alpha)
            rows.append(
                {
                    "asset": asset,
                    "window": window,
                    "alpha": alpha,
                    "model": model,
                    "n_breach": int((r < v).sum()),
                    "z1": z1(r, v, e),
                    "z1_p_approx": z1_pvalue_asymptotic(r, v, e),
                    "z2": z2v,
                    "z2_p_approx": z2p,
                    "es_optimistic": bool(z2v < 0),
                    "es_reject_approx": bool(np.isfinite(z2p) and z2p < 0.05),
                }
            )
    return pd.DataFrame(rows)


def evaluate_fz0_mcs(bt: pd.DataFrame, assets: list[str], mcs_cfg: dict, seed: int) -> pd.DataFrame:
    rows = []
    for asset, window, alpha in _grid(bt, assets):
        sub = _slice(bt, asset, window, alpha)
        var_p = _panel(sub, "var")
        es_p = _panel(sub, "es")
        models = sorted(set(var_p.columns) & set(es_p.columns))
        idx = var_p.index.intersection(es_p.index)

        # Drop days on which ANY model emitted a degenerate forecast (ES not
        # strictly negative, or a non-negative VaR). These are model-side
        # numerical artefacts; a single such row otherwise dominates a mean
        # FZ0. The drop is applied to every model so the comparison stays
        # paired. ``n_degenerate`` is reported.
        ok = np.ones(len(idx), dtype=bool)
        for m in models:
            v = var_p.loc[idx, m].to_numpy(float)
            e = es_p.loc[idx, m].to_numpy(float)
            ok &= np.isfinite(v) & np.isfinite(e) & (v < 0) & (e < -1e-6)
        n_degenerate = int((~ok).size - ok.sum())
        idx = idx[ok]

        realized = (
            sub.drop_duplicates("date").set_index("date")["realized"].reindex(idx).to_numpy(float)
        )
        losses = {
            m: fz0_loss(
                realized, var_p.loc[idx, m].to_numpy(float), es_p.loc[idx, m].to_numpy(float), alpha
            )
            for m in models
        }
        means = {m: float(np.mean(losses[m])) for m in models}
        best = min(means, key=means.__getitem__)
        mcs = model_confidence_set(
            losses,
            confidence=mcs_cfg["confidence"],
            block_len=mcs_cfg["block_bootstrap_len"],
            n_boot=mcs_cfg["n_boot"],
            seed=seed,
        )
        order = sorted(models, key=means.__getitem__)
        for rank, m in enumerate(order, 1):
            if m == best:
                dm_stat = dm_p = np.nan
            else:
                dm = diebold_mariano(losses[best], losses[m])
                dm_stat, dm_p = dm.statistic, dm.p_value
            rows.append(
                {
                    "asset": asset,
                    "window": window,
                    "alpha": alpha,
                    "model": m,
                    "n": len(idx),
                    "n_degenerate": n_degenerate,
                    "fz0_mean": means[m],
                    "fz0_rank": rank,
                    "is_best": m == best,
                    "dm_vs_best_stat": dm_stat,
                    "dm_vs_best_p": dm_p,
                    "in_mcs": m in mcs.included,
                    "mcs_p": float(mcs.p_values.get(m, np.nan)),
                }
            )
    return pd.DataFrame(rows)


def evaluate_density(bt: pd.DataFrame, assets: list[str]) -> pd.DataFrame:
    rows = []
    for asset, window, alpha in _grid(bt, assets):
        if abs(alpha - bt["alpha"].min()) > 1e-9:  # PIT is alpha-independent
            continue
        sub = _slice(bt, asset, window, alpha)
        for model, g in sub.groupby("model", observed=True):
            pit = g["pit"].to_numpy(float)
            if np.isfinite(pit).sum() < 20:
                continue
            b = berkowitz(pit)
            rows.append(
                {
                    "asset": asset,
                    "window": window,
                    "model": model,
                    "n": int(np.isfinite(pit).sum()),
                    "berkowitz_lr": b.statistic,
                    "berkowitz_p": b.p_value,
                    "mu": b.mu,
                    "rho": b.rho,
                    "sigma2": b.sigma2,
                    "reject": bool(np.isfinite(b.p_value) and b.p_value < 0.05),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #
def _headline_md(fz0: pd.DataFrame, cov: pd.DataFrame) -> str:
    out = ["# cryptorisk evaluation summary", ""]
    out += ["_Generated by `python -m cryptorisk.study.run_evaluation`._", ""]
    for (asset, window, alpha), g in fz0.groupby(["asset", "window", "alpha"], observed=True):
        out.append(f"## {asset} | window {window} | alpha {alpha:g}  (VaR {100 * (1 - alpha):g}%)")
        out.append("")
        out.append("Headline: FZ0 ranking and 90% Model Confidence Set.")
        out.append("")
        out.append("| rank | model | mean FZ0 | in MCS | MCS p | DM vs best p |")
        out.append("|---:|:--|---:|:--:|---:|---:|")
        cvg = cov.set_index(["asset", "window", "alpha", "model"])
        for _, r in g.sort_values("fz0_rank").iterrows():
            star = " **(best)**" if r["is_best"] else ""
            dmp = "-" if not np.isfinite(r["dm_vs_best_p"]) else f"{r['dm_vs_best_p']:.3f}"
            out.append(
                f"| {int(r['fz0_rank'])} | {r['model']}{star} | {r['fz0_mean']:.5f} | "
                f"{'yes' if r['in_mcs'] else 'no'} | {r['mcs_p']:.3f} | {dmp} |"
            )
        out.append("")
        try:
            passes = [
                m for m in g["model"] if bool(cvg.loc[(asset, window, alpha, m), "passes_all"])
            ]
            out.append(
                f"Pass every coverage test (Kupiec/Christoffersen/DQ): "
                f"{', '.join(passes) if passes else 'none'}."
            )
        except KeyError:
            pass
        out.append("")
    return "\n".join(out)


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets", nargs="+", default=cfg["assets"])
    ap.add_argument("--skip-volforecast", action="store_true")
    args = ap.parse_args()

    res_dir = repo_root() / cfg["paths"]["results"]
    bt = pd.read_parquet(res_dir / "backtests.parquet")
    bt["date"] = pd.to_datetime(bt["date"])

    level = cfg["evaluation"]["test_level"]
    dq_lags = cfg["evaluation"]["dq_lags"]
    mcs_cfg = cfg["evaluation"]["mcs"]
    seed = cfg["seed"]

    print("[eval] coverage ...", flush=True)
    cov = evaluate_coverage(bt, args.assets, level, dq_lags)
    cov.to_csv(res_dir / "eval_coverage.csv", index=False)

    print("[eval] expected shortfall ...", flush=True)
    es = evaluate_es(bt, args.assets)
    es.to_csv(res_dir / "eval_es.csv", index=False)

    print("[eval] FZ0 + Model Confidence Set (headline) ...", flush=True)
    fz0 = evaluate_fz0_mcs(bt, args.assets, mcs_cfg, seed)
    fz0.to_csv(res_dir / "eval_fz0_mcs.csv", index=False)

    print("[eval] density (Berkowitz) ...", flush=True)
    dens = evaluate_density(bt, args.assets)
    dens.to_csv(res_dir / "eval_density.csv", index=False)

    if not args.skip_volforecast:
        print("[eval] volatility forecast ...", flush=True)
        run_vol_forecast()

    (res_dir / "eval_summary.md").write_text(_headline_md(fz0, cov), encoding="utf-8")

    print("\n==================  HEADLINE: FZ0 ranking + 90% MCS  ==================")
    show = ["asset", "window", "alpha", "fz0_rank", "model", "fz0_mean", "in_mcs", "mcs_p"]
    print(fz0.sort_values(["asset", "window", "alpha", "fz0_rank"])[show].to_string(index=False))
    print(f"\n[eval] wrote eval_*.csv and eval_summary.md -> {res_dir}")


if __name__ == "__main__":
    main()
