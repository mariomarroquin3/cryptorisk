"""DuckDB store: schema, idempotent writes, point-in-time read."""

import numpy as np
import pandas as pd

from cryptorisk.data import store


def _ohlcv(dates, base=100.0):
    n = len(dates)
    return pd.DataFrame({
        "date": dates,
        "open": base + np.arange(n),
        "high": base + np.arange(n) + 1,
        "low": base + np.arange(n) - 1,
        "close": base + np.arange(n),
        "volume": np.full(n, 10.0),
    })


def test_connect_creates_schema(tmp_path):
    con = store.connect(tmp_path / "s.duckdb")
    counts = store.table_counts(con)
    assert set(counts) == {
        "prices_daily", "returns_daily", "bars_5m",
        "realized_daily", "context_daily", "microstructure_daily",
    }
    assert all(v == 0 for v in counts.values())
    con.close()


def test_write_prices_is_idempotent(tmp_path):
    con = store.connect(tmp_path / "s.duckdb")
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    df = _ohlcv(dates)
    store.write_prices_daily(con, "BTC", "binance", df)
    store.write_prices_daily(con, "BTC", "binance", df)  # re-run
    assert store.table_counts(con)["prices_daily"] == 10
    # a different source coexists
    store.write_prices_daily(con, "BTC", "coinmetrics", df)
    assert store.table_counts(con)["prices_daily"] == 20
    con.close()


def test_returns_roundtrip(tmp_path):
    con = store.connect(tmp_path / "s.duckdb")
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    ret = pd.DataFrame({
        "date": dates,
        "close": [100.0, 101.0, 99.0, 102.0, 103.0],
        "log_return": [np.nan, *np.diff(np.log([100.0, 101.0, 99.0, 102.0, 103.0]))],
        "source": "coinmetrics",
    })
    store.write_returns_daily(con, "ETH", ret)
    back = store.read_returns(con, "ETH")
    assert len(back) == 5
    assert back["close"].tolist() == ret["close"].tolist()
    con.close()


def test_bars_5m_incremental_upsert(tmp_path):
    con = store.connect(tmp_path / "s.duckdb")
    m1 = pd.DataFrame({
        "ts": pd.date_range("2020-01-01", periods=100, freq="5min"),
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
    })
    # m1 covers 00:00..08:15; m2 starts at 07:55 -> overlaps the last 5 bars.
    m2 = pd.DataFrame({
        "ts": pd.date_range("2020-01-01 07:55", periods=100, freq="5min"),
        "open": 2.0, "high": 2.0, "low": 2.0, "close": 2.0, "volume": 2.0,
    })
    store.write_bars_5m(con, "BTC", m1)
    store.write_bars_5m(con, "BTC", m2)
    n = store.table_counts(con)["bars_5m"]
    assert n == 100 + 100 - 5  # 5 overlapping timestamps upserted, not duplicated
    # overlapping rows took m2's value
    v = con.execute(
        "SELECT close FROM bars_5m WHERE asset='BTC' AND ts='2020-01-01 08:00'"
    ).fetchone()[0]
    assert v == 2.0
    con.close()


def test_prices_asof_respects_vintage(tmp_path):
    con = store.connect(tmp_path / "s.duckdb")
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    store.write_prices_daily(con, "BTC", "binance", _ohlcv(dates))
    # vintage_ts is "now"; a cutoff in the past sees nothing, in the future sees all
    past = store.prices_asof(con, "BTC", "binance", "2000-01-01")
    future = store.prices_asof(con, "BTC", "binance", "2100-01-01")
    assert past.empty and len(future) == 3
    con.close()
