"""Recompute realized measures from the store's cached 5-minute bars
(V2_PLAN §2, Phase 1) -- no re-fetch, no re-ingest of daily prices.

Use this after changing anything in :mod:`cryptorisk.data.realized` (e.g. a
constant fix) to refresh the ``realized_daily`` table without re-running the
whole ``make data`` (which re-fetches every bar from data.binance.vision).

    python -m cryptorisk.study.run_realized
    python -m cryptorisk.study.run_realized --assets BTC
"""

from __future__ import annotations

import argparse

from cryptorisk.config import load_config, repo_root
from cryptorisk.data import store
from cryptorisk.data.realized import realized_daily


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets", nargs="+", default=cfg["assets"])
    ap.add_argument("--db", default=str(repo_root() / cfg["paths"]["store"]))
    args = ap.parse_args()

    con = store.connect(args.db)
    cfg_r = cfg.get("realized", {})
    for asset in args.assets:
        bars = con.execute(
            "SELECT ts, close FROM bars_5m WHERE asset = ? ORDER BY ts", [asset]
        ).df()
        if bars.empty:
            print(f"[realized] {asset}: no cached 5-min bars -- skip (run `make data` first)",
                  flush=True)
            continue
        rdf = realized_daily(
            bars, subsample=cfg_r.get("subsample", True), jump_test=cfg_r.get("jump_test", "BNS")
        )
        n = store.write_realized_daily(con, asset, rdf)
        span = f"{rdf['date'].min()} -> {rdf['date'].max()}" if n else "no rows"
        print(f"[realized] {asset}: {n} days written ({span})", flush=True)
    con.close()


if __name__ == "__main__":
    main()
