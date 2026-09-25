"""Can adaptive conformal recalibration repair a miscalibrated model?

Re-runs the study's walk-forward for a few models wrapped in ACI
(:class:`cryptorisk.models.conformal.AdaptiveConformal`) and compares each with its
*raw* counterpart from ``backtests.parquet`` on the same days:

* ``conformal_backtests.parquet``  - the wrapped walk-forward (same schema as backtests)
* ``conformal_summary.csv``        - raw vs ACI: hit rate, Kupiec / Christoffersen / DQ
                                     p-values, mean FZ0 and a Diebold-Mariano test of the
                                     FZ0 difference
* ``conformal_level_path.csv``     - the adjusted tail level over time (every 5th day)

The control models (HS, GARCH-t) show whether ACI helps everything equally or mostly the
machine-learning arm. Nothing here touches the frozen study tables.

    python -m cryptorisk.study.run_conformal
    python -m cryptorisk.study.run_conformal --models RF-QR --assets BTC
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from cryptorisk.backtest.coverage import christoffersen_cc, dq_test, kupiec_pof
from cryptorisk.backtest.engine import walk_forward
from cryptorisk.backtest.scoring import diebold_mariano, fz0_loss
from cryptorisk.config import load_config, repo_root
from cryptorisk.models.conformal import AdaptiveConformal
from cryptorisk.models.registry import all_models
from cryptorisk.study.run_backtests import _load_returns, _normalize_window

DEFAULT_MODELS = ["HS", "GARCH-t", "RF-QR", "LSTM-Vol"]
WINDOW = 500


def _job(df, model, asset, alphas, oos_start, gamma_frac):
    wrapped = AdaptiveConformal(model, tuple(alphas), gamma_frac=gamma_frac)
    res = walk_forward(
        df, wrapped, alphas=alphas, asset=asset, window=WINDOW, oos_start=oos_start, refit_every=1
    )
    trace = pd.DataFrame(wrapped.trace, columns=["date", "asset", "alpha", "level"])
    trace["model"] = wrapped.name
    return _normalize_window(res.frame), trace


def _stats(g: pd.DataFrame, alpha: float, dq_lags: int) -> dict:
    viol = g["violation"].fillna(False).to_numpy(bool)
    var = g["var"].to_numpy(float)
    loss = fz0_loss(g["realized"], g["var"], g["es"], alpha)
    return {
        "n": int(viol.size),
        "hit_rate": float(viol.mean()),
        "kupiec_p": kupiec_pof(viol, alpha).p_value,
        "chr_cc_p": christoffersen_cc(viol, alpha).p_value,
        "dq_p": dq_test(viol, var, alpha, lags=dq_lags).p_value,
        "fz0_mean": float(np.mean(loss)),
        "mean_var": float(np.nanmean(var)),
        "mean_es": float(np.nanmean(g["es"])),
        "_loss": loss,
    }


def summarise(raw: pd.DataFrame, aci: pd.DataFrame, dq_lags: int = 4) -> pd.DataFrame:
    """One row per (asset, alpha, base model, variant); ACI rows carry the DM test of
    FZ0(ACI) - FZ0(raw) (negative mean = ACI scores better)."""
    rows = []
    for (asset, alpha, name), g_aci in aci.groupby(["asset", "alpha", "model"], observed=True):
        base = name.removesuffix("+ACI")
        g_raw = raw[(raw["asset"] == asset) & (raw["alpha"] == alpha) & (raw["model"] == base)]
        # same days for both, so the comparison is like for like
        common = np.intersect1d(g_raw["date"].to_numpy(), g_aci["date"].to_numpy())
        g_raw = g_raw[g_raw["date"].isin(common)].sort_values("date")
        g_aci = g_aci[g_aci["date"].isin(common)].sort_values("date")
        if g_raw.empty or len(g_raw) != len(g_aci):
            continue
        s_raw, s_aci = _stats(g_raw, alpha, dq_lags), _stats(g_aci, alpha, dq_lags)
        dm = diebold_mariano(s_aci["_loss"], s_raw["_loss"])
        for variant, s in (("raw", s_raw), ("ACI", s_aci)):
            row = {"asset": asset, "alpha": alpha, "base_model": base, "variant": variant,
                   **{k: v for k, v in s.items() if k != "_loss"}}
            if variant == "ACI":
                row.update(dm_fz0_diff=dm.mean_diff, dm_p=dm.p_value)
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", nargs="+", default=cfg["assets"])
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--gamma-frac", type=float, default=0.05, help="ACI step as a fraction of alpha")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--db", default=str(repo_root() / cfg["paths"]["store"]))
    args = ap.parse_args()

    alphas = cfg["alphas"]
    oos_start = cfg["sample"]["oos_start"]
    by_name = {m.name: m for m in all_models()}
    missing = [m for m in args.models if m not in by_name]
    if missing:
        raise SystemExit(f"unknown models {missing}")
    returns = {a: _load_returns(args.db, a) for a in args.assets}
    jobs = [(returns[a], by_name[m], a) for a in args.assets for m in args.models]
    print(f"[conformal] {len(jobs)} runs (ACI gamma = {args.gamma_frac} x alpha)", flush=True)

    out = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(_job)(df, m, a, alphas, oos_start, args.gamma_frac) for df, m, a in jobs
    )
    aci = pd.concat([f for f, _ in out], ignore_index=True)
    trace = pd.concat([t for _, t in out], ignore_index=True)

    res_dir = repo_root() / cfg["paths"]["results"]
    raw = pd.read_parquet(res_dir / "backtests.parquet")
    raw = raw[raw["window"] == WINDOW]
    aci.to_parquet(res_dir / "conformal_backtests.parquet", index=False)
    summary = summarise(raw, aci, cfg["evaluation"]["dq_lags"])
    summary.to_csv(res_dir / "conformal_summary.csv", index=False)

    path = trace.iloc[::5].reset_index(drop=True)
    path.to_csv(res_dir / "conformal_level_path.csv", index=False)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
