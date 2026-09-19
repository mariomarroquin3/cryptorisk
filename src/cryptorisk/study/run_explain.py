"""Explainability outputs for RF-QR, refit every ``--every`` OOS days (same
500-day rolling window as the backtest):

* ``explain_rf_importance.csv``  -- date, asset, feature, importance
* ``explain_rf_diagnostics.csv`` -- date, asset, ess, n_train, var_cond, var_hs
* ``explain_rf_inputs.csv``      -- latest date only: feature, value, zscore
* ``explain_lstm_importance.csv`` -- date, asset, lag (1 = yesterday), feature,
  importance: LSTM-Vol permutation importance (rise in quasi-NLL), refit
  every ``--lstm-every`` OOS days (gradient descent is costlier than a forest)

    python -m cryptorisk.study.run_explain
"""

from __future__ import annotations

import argparse

import pandas as pd
from joblib import Parallel, delayed

from cryptorisk.config import load_config, repo_root
from cryptorisk.models.base import Context
from cryptorisk.models.lstm_vol import LstmVol
from cryptorisk.models.random_forest import RandomForestQR
from cryptorisk.study.run_backtests import _load_returns

_REALIZED = ("rv", "bv", "rsv_pos", "rsv_neg", "jump", "rq")


def _run_asset(df: pd.DataFrame, asset: str, oos_start: str, window: int, every: int, alpha: float):
    d = df.sort_values("date").reset_index(drop=True)
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["log_return"].notna()].reset_index(drop=True)
    r = d["log_return"].to_numpy(float)
    dates = d["date"].to_numpy()
    start = max(window, int(d["date"].searchsorted(pd.Timestamp(oos_start))))
    model = RandomForestQR()
    imp, diag, last = [], [], None
    for k, t in enumerate(range(start, len(d) + 1)):
        # t == len(d) is the live "tomorrow" fit on the latest window
        if k % every and t != len(d):
            continue
        lo = t - window
        ctx = Context(
            returns=r[lo:t],
            dates=dates[lo:t],
            asof=dates[t - 1],
            asset=asset,
            realized={c: d[c].to_numpy(float)[lo:t] for c in _REALIZED if c in d.columns},
        )
        e = model.explain(ctx, alpha)
        if e is None:
            continue
        day = pd.Timestamp(dates[t - 1])
        for f, v in zip(e["features"], e["importance"], strict=True):
            imp.append({"date": day, "asset": asset, "feature": f, "importance": v})
        diag.append({
            "date": day, "asset": asset, "ess": e["ess"], "n_train": e["n_train"],
            "var_cond": e["var_cond"], "var_hs": e["var_hs"],
        })
        last = [
            {"asset": asset, "date": day, "feature": f, "value": v, "zscore": z}
            for f, v, z in zip(e["features"], e["inputs"], e["zscores"], strict=True)
        ]
    return imp, diag, last or []


def _run_lstm(df: pd.DataFrame, asset: str, oos_start: str, window: int, every: int):
    d = df.sort_values("date").reset_index(drop=True)
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["log_return"].notna()].reset_index(drop=True)
    r = d["log_return"].to_numpy(float)
    dates = d["date"].to_numpy()
    start = max(window, int(d["date"].searchsorted(pd.Timestamp(oos_start))))
    model, rows = LstmVol(), []
    for k, t in enumerate(range(start, len(d) + 1)):
        if k % every and t != len(d):
            continue
        ctx = Context(returns=r[t - window : t], dates=dates[t - window : t], asof=dates[t - 1], asset=asset)
        e = model.explain(ctx)
        if e is None:
            continue
        day = pd.Timestamp(dates[t - 1])
        for lag, row in enumerate(e["importance"], start=1):
            for f, v in zip(e["features"], row, strict=True):
                rows.append({"date": day, "asset": asset, "lag": lag, "feature": f, "importance": v})
    return rows


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets", nargs="+", default=cfg["assets"])
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--lstm-every", type=int, default=60)
    ap.add_argument("--skip-rf", action="store_true")
    ap.add_argument("--skip-lstm", action="store_true")
    ap.add_argument("--alpha", type=float, default=cfg["alphas"][0])
    ap.add_argument("--db", default=str(repo_root() / cfg["paths"]["store"]))
    args = ap.parse_args()

    res = repo_root() / "data" / "results"
    if not args.skip_rf:
        out = Parallel(n_jobs=-1, verbose=5)(
            delayed(_run_asset)(
                _load_returns(args.db, a), a, cfg["sample"]["oos_start"], args.window, args.every, args.alpha
            )
            for a in args.assets
        )
        pd.DataFrame([x for o in out for x in o[0]]).to_csv(res / "explain_rf_importance.csv", index=False)
        pd.DataFrame([x for o in out for x in o[1]]).to_csv(res / "explain_rf_diagnostics.csv", index=False)
        pd.DataFrame([x for o in out for x in o[2]]).to_csv(res / "explain_rf_inputs.csv", index=False)
    if not args.skip_lstm:
        lo = Parallel(n_jobs=-1, verbose=5)(
            delayed(_run_lstm)(
                _load_returns(args.db, a), a, cfg["sample"]["oos_start"], args.window, args.lstm_every
            )
            for a in args.assets
        )
        pd.DataFrame([x for o in lo for x in o]).to_csv(res / "explain_lstm_importance.csv", index=False)
    print("[explain] wrote explain_rf_*.csv / explain_lstm_importance.csv")


if __name__ == "__main__":
    main()
