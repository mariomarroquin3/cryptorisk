"""DuckDB store (V2_PLAN §2).

One file, no server. Tables:

* ``prices_daily(asset, date, source, open, high, low, close, volume, vintage_ts)``
* ``returns_daily(asset, date, close, log_return, source)``  -- reconciled canonical series
* ``bars_5m(asset, ts, open, high, low, close, volume)``
* ``realized_daily(asset, date, rv, bv, rsv_pos, rsv_neg, jump, n_bars)``
* ``context_daily(date, spx, dxy, fed_funds, cpi_lag, hashrate, difficulty)``
* ``microstructure_daily(asset, date, funding_8h, open_interest, netflow, stbl_supply_chg)``

**Point-in-time:** ``prices_daily`` rows carry ``vintage_ts`` (when the row was
pulled). A backtest as of date *t* should read the newest vintage with
``vintage_ts <= t`` -- helper :func:`prices_asof`.

All writers are idempotent: they ``DELETE`` the (asset, date-range, source) slice
they are about to write, then ``INSERT``. Re-running ingestion never duplicates.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd

SCHEMA_VERSION = 2

_DDL = f"""
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
INSERT OR REPLACE INTO meta VALUES ('schema_version', '{SCHEMA_VERSION}');

CREATE TABLE IF NOT EXISTS prices_daily (
    asset      TEXT    NOT NULL,
    date       DATE    NOT NULL,
    source     TEXT    NOT NULL,
    open       DOUBLE,
    high       DOUBLE,
    low        DOUBLE,
    close      DOUBLE  NOT NULL,
    volume     DOUBLE,
    vintage_ts TIMESTAMP NOT NULL,
    PRIMARY KEY (asset, date, source)
);

CREATE TABLE IF NOT EXISTS returns_daily (
    asset      TEXT   NOT NULL,
    date       DATE   NOT NULL,
    close      DOUBLE NOT NULL,
    log_return DOUBLE,
    source     TEXT   NOT NULL,
    PRIMARY KEY (asset, date)
);

CREATE TABLE IF NOT EXISTS bars_5m (
    asset  TEXT      NOT NULL,
    ts     TIMESTAMP NOT NULL,
    open   DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE NOT NULL, volume DOUBLE,
    PRIMARY KEY (asset, ts)
);

CREATE TABLE IF NOT EXISTS realized_daily (
    asset   TEXT   NOT NULL,
    date    DATE   NOT NULL,
    rv      DOUBLE, bv DOUBLE, rsv_pos DOUBLE, rsv_neg DOUBLE, jump DOUBLE, rq DOUBLE,
    n_bars  INTEGER,
    PRIMARY KEY (asset, date)
);
ALTER TABLE realized_daily ADD COLUMN IF NOT EXISTS rq DOUBLE;

CREATE TABLE IF NOT EXISTS context_daily (
    date       DATE PRIMARY KEY,
    spx        DOUBLE, dxy DOUBLE, fed_funds DOUBLE, cpi_lag DOUBLE,
    hashrate   DOUBLE, difficulty DOUBLE
);

CREATE TABLE IF NOT EXISTS microstructure_daily (
    asset            TEXT NOT NULL,
    date             DATE NOT NULL,
    funding_8h       DOUBLE, open_interest DOUBLE, netflow DOUBLE, stbl_supply_chg DOUBLE,
    PRIMARY KEY (asset, date)
);
"""


def connect(path: str | Path, *, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(p), read_only=read_only)
    if not read_only:
        con.execute(_DDL)
    return con


def now_utc() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _replace(con, table: str, df: pd.DataFrame, keys: dict[str, object]) -> int:
    """Delete the slice identified by ``keys`` (equality) then insert ``df``."""
    if df.empty:
        return 0
    where = " AND ".join(f"{k} = ?" for k in keys)
    con.execute(f"DELETE FROM {table} WHERE {where}", list(keys.values()))
    con.register("_incoming", df)
    cols = ", ".join(df.columns)
    con.execute(f"INSERT INTO {table} ({cols}) SELECT {cols} FROM _incoming")
    con.unregister("_incoming")
    return len(df)


def write_prices_daily(con, asset: str, source: str, df: pd.DataFrame) -> int:
    """``df`` columns: date, open, high, low, close, volume."""
    out = df.copy()
    out["asset"] = asset
    out["source"] = source
    out["vintage_ts"] = now_utc()
    out = out[["asset", "date", "source", "open", "high", "low", "close", "volume", "vintage_ts"]]
    return _replace(con, "prices_daily", out, {"asset": asset, "source": source})


def write_bars_5m(con, asset: str, df: pd.DataFrame) -> int:
    """Upsert on (asset, ts). Safe to call incrementally, month by month."""
    if df.empty:
        return 0
    out = df.copy()
    out["asset"] = asset
    out = out[["asset", "ts", "open", "high", "low", "close", "volume"]]
    con.register("_incoming", out)
    con.execute("INSERT OR REPLACE INTO bars_5m SELECT * FROM _incoming")
    con.unregister("_incoming")
    return len(out)


def write_realized_daily(con, asset: str, df: pd.DataFrame) -> int:
    out = df.copy()
    out["asset"] = asset
    out = out[["asset", "date", "rv", "bv", "rsv_pos", "rsv_neg", "jump", "rq", "n_bars"]]
    return _replace(con, "realized_daily", out, {"asset": asset})


def write_returns_daily(con, asset: str, df: pd.DataFrame) -> int:
    """``df`` columns: date, close, log_return, source."""
    out = df.copy()
    out["asset"] = asset
    out = out[["asset", "date", "close", "log_return", "source"]]
    return _replace(con, "returns_daily", out, {"asset": asset})


def write_context_daily(con, df: pd.DataFrame) -> int:
    con.execute("DELETE FROM context_daily")
    con.register("_incoming", df)
    con.execute("INSERT INTO context_daily SELECT * FROM _incoming")
    con.unregister("_incoming")
    return len(df)


def write_microstructure_daily(con, asset: str, df: pd.DataFrame) -> int:
    out = df.copy()
    out["asset"] = asset
    out = out[["asset", "date", "funding_8h", "open_interest", "netflow", "stbl_supply_chg"]]
    return _replace(con, "microstructure_daily", out, {"asset": asset})


def read_returns(con, asset: str) -> pd.DataFrame:
    return con.execute(
        "SELECT date, close, log_return FROM returns_daily WHERE asset = ? ORDER BY date",
        [asset],
    ).df()


def prices_asof(con, asset: str, source: str, asof: str | pd.Timestamp) -> pd.DataFrame:
    """Point-in-time read: rows whose vintage is not after ``asof``."""
    return con.execute(
        """
        SELECT date, open, high, low, close, volume
        FROM prices_daily
        WHERE asset = ? AND source = ? AND vintage_ts <= ?
        ORDER BY date
        """,
        [asset, source, pd.Timestamp(asof)],
    ).df()


def table_counts(con) -> dict[str, int]:
    tables = [
        "prices_daily", "returns_daily", "bars_5m", "realized_daily",
        "context_daily", "microstructure_daily",
    ]
    return {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in tables}
