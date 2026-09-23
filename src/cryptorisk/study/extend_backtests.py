"""Roll the walk-forward forward past the frozen study sample, store-free.

The study's ``data/results/backtests.parquet`` is frozen at the sample end
(``2026-09-10``) so every evaluation table stays reproducible. This appends the
out-of-sample days since then -- same models, same 500-day / expanding windows,
same engine -- to a *separate* file, ``data/results/backtests_live.parquet``,
which only the API reads (the Overview band, "latest VaR by model"). Nothing in
the evaluation pipeline sees it.

Inputs come from ``price_history.parquet`` (rolled forward first, from Binance
5-min klines, realized measures included), so it runs anywhere with the ``api``
+ ``ml`` extras -- no DuckDB store. MS-GARCH is left out: its walk-forward runs
in R and is served from the store.

    python -m cryptorisk.study.extend_backtests
"""

from __future__ import annotations

import argparse

import pandas as pd
from joblib import Parallel, delayed

from cryptorisk.backtest.engine import walk_forward
from cryptorisk.config import load_config, repo_root
from cryptorisk.models.registry import all_models
from cryptorisk.study.refresh_price_history import refresh
from cryptorisk.study.run_backtests import EXPANDING_WINDOW, _normalize_window

LIVE_FILE = "backtests_live.parquet"
_SKIP = {"MS-GARCH"}
_KEY = ["model", "asset", "window", "alpha", "date"]


def _job(df, model, asset, window, alphas, start, refit_every):
    try:
        res = walk_forward(
            df, model, alphas=alphas, asset=asset, window=window, oos_start=start,
            refit_every=refit_every,
        )
    except Exception as exc:  # one broken model must not sink the daily refresh
        print(f"[extend] {model.name}/{asset}/{window}: skipped ({type(exc).__name__}: {exc})", flush=True)
        return pd.DataFrame()
    return _normalize_window(res.frame) if not res.frame.empty else pd.DataFrame()


def extend(*, n_jobs: int = -1, refresh_prices: bool = True) -> pd.DataFrame:
    cfg = load_config()
    res_dir = repo_root() / cfg["paths"]["results"]
    hist = refresh() if refresh_prices else pd.read_parquet(res_dir / "price_history.parquet")
    hist["date"] = pd.to_datetime(hist["date"])

    frozen = pd.read_parquet(res_dir / "backtests.parquet", columns=["asset", "window", "date"])
    live_path = res_dir / LIVE_FILE
    live = pd.read_parquet(live_path) if live_path.exists() else pd.DataFrame()
    have = pd.concat([frozen, live[["asset", "window", "date"]]] if not live.empty else [frozen])
    last_done = have.groupby(["asset", "window"])["date"].max()

    windows = [*cfg["walk_forward"]["windows"], *([EXPANDING_WINDOW] if cfg["walk_forward"].get("expanding") else [])]
    refit_map = cfg["walk_forward"]["refit_every"]
    default_refit = refit_map.get("default", 1)
    models = [m for m in all_models() if m.name not in _SKIP]

    jobs = []
    for asset, g in hist.groupby("asset"):
        if asset not in cfg["assets"]:
            continue
        frame = g.drop(columns="asset").sort_values("date")
        for w in windows:
            done = last_done.get((asset, w))
            if done is None or frame["date"].max() <= done:
                continue
            start = done + pd.Timedelta(days=1)
            win = "expanding" if w == EXPANDING_WINDOW else int(w)
            jobs += [(frame, m, asset, win, start) for m in models]
    if not jobs:
        print("[extend] already up to date", flush=True)
        return live
    print(f"[extend] {len(jobs)} runs", flush=True)

    frames = Parallel(n_jobs=n_jobs)(
        delayed(_job)(f, m, a, w, cfg["alphas"], s, refit_map.get(m.name, default_refit))
        for f, m, a, w, s in jobs
    )
    new = [f for f in frames if not f.empty]
    if not new:
        return live
    out = pd.concat([live, *new], ignore_index=True) if not live.empty else pd.concat(new, ignore_index=True)
    out = out.drop_duplicates(_KEY, keep="last").sort_values(_KEY).reset_index(drop=True)
    out.to_parquet(live_path, index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--no-refresh", action="store_true", help="do not roll price_history forward first")
    args = ap.parse_args()
    out = extend(n_jobs=args.n_jobs, refresh_prices=not args.no_refresh)
    if out.empty:
        print("[extend] nothing written")
        return
    last = out.groupby("asset")["date"].max().dt.date.to_dict()
    print(f"[extend] {len(out):,} live rows; last dates {last}")


if __name__ == "__main__":
    main()
