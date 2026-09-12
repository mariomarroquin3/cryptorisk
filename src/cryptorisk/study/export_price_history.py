"""Export a compact price-history snapshot for the API (`cryptorisk.api`).

The API's own docstring says it's meant to run off `data/results/` alone,
but `api/data.py::load_price_window`/`regime_series` used to query the
DuckDB store directly -- fine locally, wrong for deploy: that store carries
5-min bars the API never touches and is 100+MB, gitignored, and not
something a host like Render (no free persistent disk) can be handed.

This writes just the daily columns `/prices/{asset}` and the live
`/forecast/{asset}` re-fit actually read to `data/results/price_history.parquet`
(a few hundred KB) so the whole API can ship from a `data/results/` snapshot
committed to git. Re-run after `make data && make realized` whenever the
store is refreshed.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from cryptorisk.config import load_config, repo_root


def export_price_history() -> pd.DataFrame:
    cfg = load_config()
    store = repo_root() / cfg["paths"]["store"]
    con = duckdb.connect(str(store), read_only=True)
    try:
        df = con.execute(
            """
            SELECT r.asset, r.date, r.close, r.log_return,
                   x.rv, x.bv, x.rsv_pos, x.rsv_neg, x.jump, x.rq
            FROM returns_daily r
            LEFT JOIN realized_daily x USING (asset, date)
            WHERE r.asset = ANY(?)
            ORDER BY r.asset, r.date
            """,
            [cfg["assets"]],
        ).df()
    finally:
        con.close()
    out = repo_root() / cfg["paths"]["results"] / "price_history.parquet"
    df.to_parquet(out, index=False)
    return df


if __name__ == "__main__":
    result = export_price_history()
    assets = sorted(result["asset"].unique())
    print(f"-> data/results/price_history.parquet: {len(result)} rows, assets={assets}")
