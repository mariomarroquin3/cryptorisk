"""Run the walk-forward for every (model, asset, window) and persist tidy
results to ``data/results/backtests.parquet`` (V2_PLAN §8, Phase 3 uses these).

    python -m cryptorisk.study.run_backtests
    python -m cryptorisk.study.run_backtests --models HS GARCH-t --assets BTC
"""

from __future__ import annotations

import argparse

import duckdb
import pandas as pd
from joblib import Parallel, delayed

from cryptorisk.backtest.engine import walk_forward
from cryptorisk.config import load_config, repo_root
from cryptorisk.models.registry import all_models


def _load_returns(db: str, asset: str) -> pd.DataFrame:
    con = duckdb.connect(db, read_only=True)
    r = con.execute(
        "SELECT date, log_return FROM returns_daily WHERE asset = ? ORDER BY date", [asset]
    ).df()
    rlz = con.execute(
        "SELECT date, rv, bv, rsv_pos, rsv_neg, jump FROM realized_daily WHERE asset = ? ORDER BY date",
        [asset],
    ).df()
    con.close()
    return r.merge(rlz, on="date", how="left")


def _job(df, model, asset, window, alphas, oos_start):
    res = walk_forward(
        df, model, alphas=alphas, asset=asset, window=window, oos_start=oos_start
    )
    return res.frame


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets", nargs="+", default=cfg["assets"])
    ap.add_argument("--models", nargs="+", default=None, help="subset by model name")
    ap.add_argument("--windows", nargs="+", type=str, default=None)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--db", default=str(repo_root() / cfg["paths"]["store"]))
    args = ap.parse_args()

    alphas = cfg["alphas"]
    oos_start = cfg["sample"]["oos_start"]
    windows: list = args.windows or list(cfg["walk_forward"]["windows"])
    if cfg["walk_forward"].get("expanding") and not args.windows:
        windows = [*windows, "expanding"]
    windows = [int(w) if str(w).isdigit() else w for w in windows]

    models = all_models()
    if args.models:
        wanted = set(args.models)
        models = [m for m in models if m.name in wanted]
        if not models:
            raise SystemExit(f"no models matched {sorted(wanted)}")

    returns = {a: _load_returns(args.db, a) for a in args.assets}

    jobs = [
        (returns[a], m, a, w)
        for a in args.assets
        for m in models
        for w in windows
    ]
    print(f"[backtests] {len(jobs)} runs: {len(models)} models x {len(args.assets)} assets "
          f"x {len(windows)} windows | OOS from {oos_start}", flush=True)

    frames = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(_job)(df, m, a, w, alphas, oos_start) for df, m, a, w in jobs
    )
    out = pd.concat(frames, ignore_index=True)

    res_dir = repo_root() / "data" / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    pq = res_dir / "backtests.parquet"
    if args.models and pq.exists():
        # partial re-run: keep other models' rows
        prev = pd.read_parquet(pq)
        prev = prev[~prev["model"].isin({m.name for m in models})]
        out = pd.concat([prev, out], ignore_index=True)
    out.to_parquet(pq, index=False)

    summ = (
        out.groupby(["model", "asset", "window", "alpha"], observed=True)
        .agg(n=("violation", "size"),
             hit_rate=("violation", "mean"),
             var_avg=("var", "mean"),
             es_avg=("es", "mean"))
        .reset_index()
    )
    summ.to_csv(res_dir / "backtests_summary.csv", index=False)
    print(f"[backtests] wrote {len(out):,} rows -> data/results/backtests.parquet")
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
