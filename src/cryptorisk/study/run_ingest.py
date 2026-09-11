"""Ingest every data block into the store, then run quality checks
(V2_PLAN §8, Phase 1).

    python -m cryptorisk.study.run_ingest                 # everything
    python -m cryptorisk.study.run_ingest --skip-intraday # daily + context only
    python -m cryptorisk.study.run_ingest --quality-only  # re-run checks on the store
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


def _ingest_daily(con, asset, start, end, per_asset_daily, per_asset_sources) -> None:
    ref_name, c = prices_daily.fetch_reference_daily(asset, start, end)
    _log(f"{asset}: daily prices (Binance + {ref_name})")
    b = prices_daily.fetch_binance_daily(asset, start, end)
    store.write_prices_daily(con, asset, "binance", b)
    store.write_prices_daily(
        con, asset, ref_name,
        c.assign(open=pd.NA, high=pd.NA, low=pd.NA, volume=pd.NA)[
            ["date", "open", "high", "low", "close", "volume"]
        ],
    )
    ret = prices_daily.build_returns(b, c, reference_name=ref_name)
    store.write_returns_daily(con, asset, ret)
    per_asset_daily[asset] = ret.merge(b[["date", "volume"]], on="date", how="left")
    per_asset_sources[asset] = (b[["date", "close"]], c[["date", "close"]])
    _log(f"{asset}: {len(ret)} daily returns {ret['date'].min().date()} -> {ret['date'].max().date()}")


def run(
    assets: list[str],
    start: str,
    end: str | None,
    *,
    skip_intraday: bool,
    db_path: str,
    daily_only: list[str] = (),
) -> None:
    """``assets`` get the full treatment (daily + 5-min bars + realized);
    ``daily_only`` (the portfolio basket's extra assets) get only
    ``returns_daily`` -- the copula marginals are plain GARCH-t and the
    ``Direct-*`` models are return-only, so that is all ``make portfolio``
    needs."""
    con = store.connect(db_path)
    per_asset_daily: dict[str, pd.DataFrame] = {}
    per_asset_sources: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    per_asset_realized: dict[str, pd.DataFrame] = {}
    all_assets = [*assets, *[a for a in daily_only if a not in assets]]

    for asset in all_assets:
        _ingest_daily(con, asset, start, end, per_asset_daily, per_asset_sources)

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

    for asset in all_assets:
        _log(f"{asset}: microstructure (funding, OI - partial)")
        store.write_microstructure_daily(con, asset, microstructure.build_microstructure(asset, start, end))

    _log(f"store row counts: {store.table_counts(con)}")
    run_quality(con, per_asset_daily, per_asset_sources, per_asset_realized)
    con.close()


def _read_quality_frames(con, assets: list[str]):
    per_daily, per_src, per_rlz = {}, {}, {}
    for a in assets:
        ret = con.execute(
            "SELECT date, close, log_return FROM returns_daily WHERE asset = ? ORDER BY date", [a]
        ).df()
        vol = con.execute(
            "SELECT date, volume FROM prices_daily WHERE asset = ? AND source = 'binance' ORDER BY date", [a]
        ).df()
        per_daily[a] = ret.merge(vol, on="date", how="left")
        pa = con.execute(
            "SELECT date, close FROM prices_daily WHERE asset = ? AND source = 'binance' ORDER BY date", [a]
        ).df()
        pb = con.execute(
            "SELECT date, close FROM prices_daily WHERE asset = ? AND source <> 'binance' ORDER BY date", [a]
        ).df()
        per_src[a] = (pa, pb)
        per_rlz[a] = con.execute(
            "SELECT date, n_bars FROM realized_daily WHERE asset = ? ORDER BY date", [a]
        ).df()
    return per_daily, per_src, per_rlz


def run_quality(con, per_asset_daily, per_asset_sources, per_asset_realized) -> None:
    _log("quality checks")
    report = quality.check_all(per_asset_daily, per_asset_sources or None, per_asset_realized or None)
    rules = quality.load_allowlist(repo_root() / "config" / "quality_allowlist.yaml")
    unexplained, explained = quality.apply_allowlist(report, rules)

    out_dir = repo_root() / "data" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_dir / "quality_report.csv", index=False)
    unexplained.to_csv(out_dir / "quality_report_unexplained.csv", index=False)
    explained.to_csv(out_dir / "quality_report_explained.csv", index=False)

    _log(f"quality: {len(report)} flags total | {len(explained)} allow-listed | "
         f"{len(unexplained)} UNEXPLAINED")
    if len(report):
        _log("\n" + report["kind"].value_counts().to_string())
    if len(unexplained):
        _log("\nUNEXPLAINED (must be zero for Phase 1 done):\n" + unexplained.to_string(index=False))
        raise SystemExit(1)


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets", nargs="+", default=cfg["assets"])
    ap.add_argument("--start", default=cfg["sample"]["start"])
    ap.add_argument("--end", default=cfg["sample"]["end"])
    ap.add_argument("--skip-intraday", action="store_true")
    ap.add_argument("--skip-portfolio-extras", action="store_true",
                    help="do not also daily-ingest config.portfolio.assets not in --assets")
    ap.add_argument("--quality-only", action="store_true",
                    help="re-run quality checks against the existing store, no fetching")
    ap.add_argument("--db", default=str(repo_root() / cfg["paths"]["store"]))
    args = ap.parse_args()
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)

    # The portfolio basket (config.portfolio.assets) needs returns_daily for
    # assets outside the single-asset study universe. Ingest them daily-only
    # unless the caller narrowed --assets or opted out, so `make data` alone
    # sets up everything `make portfolio` reads.
    extras: list[str] = []
    if not args.skip_portfolio_extras and args.assets == cfg["assets"]:
        pa = (cfg.get("portfolio") or {}).get("assets", [])
        extras = [a for a in pa if a not in args.assets]

    if args.quality_only:
        con = store.connect(args.db, read_only=True)
        run_quality(con, *_read_quality_frames(con, [*args.assets, *extras]))
        con.close()
        return
    run(args.assets, args.start, args.end, skip_intraday=args.skip_intraday,
        db_path=args.db, daily_only=extras)


if __name__ == "__main__":
    main()
