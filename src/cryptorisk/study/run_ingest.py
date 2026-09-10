"""Ingest every data block into the store, then run quality checks
(V2_PLAN §8, Phase 1).

    python -m cryptorisk.study.run_ingest                 # everything
    python -m cryptorisk.study.run_ingest --skip-intraday # daily + context only
    python -m cryptorisk.study.run_ingest --assets BTC --start 2020-01-01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cryptorisk.config import load_config, repo_root
from cryptorisk.data import quality, store
from cryptorisk.data.ingest import binance_klines, context, microstructure, prices_daily
from cryptorisk.data.realized import realized_daily


def _log(msg: str) -> None:
    print(f"[ingest] {msg}", flush=True)


def run(assets: list[str], start: str, end: str | None, *, skip_intraday: bool, db_path: str) -> None:
    con = store.connect(db_path)
    per_asset_daily: dict[str, pd.DataFrame] = {}
    per_asset_sources: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    per_asset_realized: dict[str, pd.DataFrame] = {}

    for asset in assets:
        _log(f"{asset}: daily prices (Binance + CoinMetrics)")
        b = prices_daily.fetch_binance_daily(asset, start, end)
        c = prices_daily.fetch_coinmetrics_daily(asset, start, end)
        store.write_prices_daily(con, asset, "binance", b)
        store.write_prices_daily(
            con, asset, "coinmetrics",
            c.assign(open=pd.NA, high=pd.NA, low=pd.NA, volume=pd.NA)[
                ["date", "open", "high", "low", "close", "volume"]
            ],
        )
        ret = prices_daily.build_returns(b, c)
        store.write_returns_daily(con, asset, ret)
        per_asset_daily[asset] = ret.merge(b[["date", "volume"]], on="date", how="left")
        per_asset_sources[asset] = (b[["date", "close"]], c[["date", "close"]])
        _log(f"{asset}: {len(ret)} daily returns {ret['date'].min().date()} -> {ret['date'].max().date()}")

    if not skip_intraday:
        for asset in assets:
            _log(f"{asset}: 5-minute bars (data.binance.vision, monthly)")
            n_bars = 0
            for month, bars in binance_klines.iter_binance_5m(asset, start, end):
                store.write_bars_5m(con, asset, bars)
                n_bars += len(bars)
                _log(f"  {asset} {month.year}-{month.month:02d}: {len(bars)} bars (cum {n_bars})")
            all_bars = con.execute(
                "SELECT ts, close FROM bars_5m WHERE asset = ? ORDER BY ts", [asset]
            ).df()
            if not all_bars.empty:
                cfg_r = load_config().get("realized", {})
                rdf = realized_daily(
                    all_bars,
                    subsample=cfg_r.get("subsample", True),
                    jump_test=cfg_r.get("jump_test", "BNS"),
                )
                store.write_realized_daily(con, asset, rdf)
                per_asset_realized[asset] = rdf
                _log(f"{asset}: realized measures for {len(rdf)} days")

    _log("context (hashrate/difficulty, SPX/DXY, fed/CPI)")
    store.write_context_daily(con, context.build_context(start, end))

    for asset in assets:
        _log(f"{asset}: microstructure (funding, OI - partial)")
        store.write_microstructure_daily(con, asset, microstructure.build_microstructure(asset, start, end))

    _log("quality checks")
    report = quality.check_all(per_asset_daily, per_asset_sources or None, per_asset_realized or None)
    out_dir = repo_root() / "data" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_dir / "quality_report.csv", index=False)

    counts = store.table_counts(con)
    con.close()

    _log(f"store row counts: {counts}")
    if report.empty:
        _log("quality: no flags")
    else:
        _log(f"quality: {len(report)} flags -> data/results/quality_report.csv")
        _log("\n" + report["kind"].value_counts().to_string())


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets", nargs="+", default=cfg["assets"])
    ap.add_argument("--start", default=cfg["sample"]["start"])
    ap.add_argument("--end", default=cfg["sample"]["end"])
    ap.add_argument("--skip-intraday", action="store_true")
    ap.add_argument("--db", default=str(repo_root() / cfg["paths"]["store"]))
    args = ap.parse_args()
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    run(args.assets, args.start, args.end, skip_intraday=args.skip_intraday, db_path=args.db)


if __name__ == "__main__":
    main()
