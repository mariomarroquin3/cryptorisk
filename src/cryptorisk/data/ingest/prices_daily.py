"""Daily prices from two independent sources, reconciled (V2_PLAN §2).

* **Binance** spot klines (``BTCUSDT`` / ``ETHUSDT``): exchange OHLCV.
* **CoinMetrics** community ``PriceUSD``: a multi-venue reference rate,
  methodologically independent of any single exchange.

The canonical return series uses the CoinMetrics reference close (falls back to
Binance where CM is missing). Both sources are stored in ``prices_daily`` for
the divergence check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptorisk.data._http import get_json

_BINANCE_SYMBOL = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}
_CM_ASSET = {"BTC": "btc", "ETH": "eth"}
_BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
_CM_METRICS = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"


def fetch_binance_daily(asset: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Columns: date, open, high, low, close, volume."""
    sym = _BINANCE_SYMBOL[asset]
    t0 = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    t1 = int((pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.now(tz="UTC")).timestamp() * 1000)
    rows: list[list] = []
    cursor = t0
    while cursor < t1:
        batch = get_json(
            _BINANCE_KLINES,
            params={"symbol": sym, "interval": "1d", "startTime": cursor, "endTime": t1, "limit": 1000},
        )
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0] + 86_400_000
        if len(batch) < 1000:
            break
    if not rows:
        return _empty_ohlcv()
    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "qav", "trades", "tbb", "tbq", "ignore",
    ])
    df["date"] = pd.to_datetime(df["open_time"], unit="ms").dt.normalize()
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c])
    return (
        df[["date", "open", "high", "low", "close", "volume"]]
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )


def fetch_coinmetrics_daily(asset: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Columns: date, close  (PriceUSD reference rate)."""
    params = {
        "assets": _CM_ASSET[asset],
        "metrics": "PriceUSD",
        "frequency": "1d",
        "start_time": start,
        "page_size": 10_000,
    }
    if end:
        params["end_time"] = end
    url, rows = _CM_METRICS, []
    while True:
        payload = get_json(url, params=params if url == _CM_METRICS else None)
        rows.extend(payload.get("data", []))
        nxt = payload.get("next_page_url")
        if not nxt:
            break
        url = nxt
    if not rows:
        return pd.DataFrame(columns=["date", "close"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["time"]).dt.tz_localize(None).dt.normalize()
    df["close"] = pd.to_numeric(df["PriceUSD"])
    return df[["date", "close"]].drop_duplicates("date").sort_values("date").reset_index(drop=True)


def build_returns(binance: pd.DataFrame, coinmetrics: pd.DataFrame) -> pd.DataFrame:
    """Canonical series. Columns: date, close, log_return, source.

    ``close`` = CoinMetrics reference price where available, else Binance close.
    """
    b = binance[["date", "close"]].rename(columns={"close": "close_binance"})
    c = coinmetrics.rename(columns={"close": "close_cm"})
    m = c.merge(b, on="date", how="outer").sort_values("date").reset_index(drop=True)
    m["close"] = m["close_cm"].where(m["close_cm"].notna(), m["close_binance"])
    m["source"] = np.where(m["close_cm"].notna(), "coinmetrics", "binance")
    m = m.dropna(subset=["close"]).reset_index(drop=True)
    m["log_return"] = np.log(m["close"]).diff()
    return m[["date", "close", "log_return", "source"]]


def _empty_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
