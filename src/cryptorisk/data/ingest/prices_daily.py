"""Daily prices from two independent sources, reconciled (V2_PLAN §2).

* **Binance** spot klines (``BTCUSDT`` / ``ETHUSDT`` / ...): exchange OHLCV,
  quoted in USDT.
* A **reference rate**, methodologically independent of Binance:
  - **CoinMetrics** community ``PriceUSD`` (multi-venue) for BTC / ETH / BNB;
  - **Coinbase** ``SOL-USD`` daily candles (independent USD venue) for SOL,
    because CoinMetrics' community tier does not serve SOL.
  :func:`fetch_reference_daily` dispatches per asset and returns the provider
  name so the store keeps the true label.

The canonical return series uses the reference close (falls back to Binance
where the reference is missing). Both sources are stored in ``prices_daily`` for
the divergence check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptorisk.data._http import get_json

_BINANCE_SYMBOL = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "BNB": "BNBUSDT"}
_CM_ASSET = {"BTC": "btc", "ETH": "eth", "SOL": "sol", "BNB": "bnb"}
_COINBASE_PRODUCT = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "BNB": "BNB-USD"}
# reference-rate provider per asset (CoinMetrics community does not serve SOL)
_REFERENCE = {"BTC": "coinmetrics", "ETH": "coinmetrics", "BNB": "coinmetrics", "SOL": "coinbase"}
_BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
_CM_METRICS = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
_COINBASE_CANDLES = "https://api.exchange.coinbase.com/products/{product}/candles"


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
    """Columns: date, close  (PriceUSD reference rate). Returns an empty frame if
    the community tier does not serve this asset (HTTP 403) rather than raising,
    so the caller can fall back to another source."""
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
        try:
            payload = get_json(url, params=params if url == _CM_METRICS else None)
        except RuntimeError as exc:  # 403 for assets outside the community tier
            if "403" in str(exc):
                return pd.DataFrame(columns=["date", "close"])
            raise
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


def fetch_coinbase_daily(asset: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Columns: date, close  (Coinbase Exchange daily candle close, USD).

    Coinbase caps a candle request at 300 rows, so walk the range in ~290-day
    windows. Candle row layout is ``[time, low, high, open, close, volume]`` with
    ``time`` in epoch seconds.
    """
    url = _COINBASE_CANDLES.format(product=_COINBASE_PRODUCT[asset])
    lo = pd.Timestamp(start, tz="UTC")
    hi_end = pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.now(tz="UTC")
    step = pd.Timedelta(days=290)
    rows: list[list] = []
    while lo < hi_end:
        hi = min(lo + step, hi_end)
        batch = get_json(url, params={
            "granularity": 86400,
            "start": lo.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": hi.strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
        if batch:
            rows.extend(batch)
        lo = hi
    if not rows:
        return pd.DataFrame(columns=["date", "close"])
    df = pd.DataFrame(rows, columns=["t", "low", "high", "open", "close", "volume"])
    df["date"] = pd.to_datetime(pd.to_numeric(df["t"]), unit="s").dt.normalize()
    df["close"] = pd.to_numeric(df["close"])
    return df[["date", "close"]].drop_duplicates("date").sort_values("date").reset_index(drop=True)


def fetch_reference_daily(
    asset: str, start: str, end: str | None = None
) -> tuple[str, pd.DataFrame]:
    """Return ``(provider_name, frame)`` for the asset's reference rate.
    ``frame`` columns: date, close."""
    provider = _REFERENCE.get(asset, "coinmetrics")
    if provider == "coinbase":
        return "coinbase", fetch_coinbase_daily(asset, start, end)
    return "coinmetrics", fetch_coinmetrics_daily(asset, start, end)


def build_returns(
    binance: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    reference_name: str = "coinmetrics",
) -> pd.DataFrame:
    """Canonical series. Columns: date, close, log_return, source.

    ``close`` = reference price where available, else Binance close. ``source``
    records which provider each day came from.
    """
    b = binance[["date", "close"]].rename(columns={"close": "close_binance"})
    c = reference[["date", "close"]].rename(columns={"close": "close_ref"})
    m = c.merge(b, on="date", how="outer").sort_values("date").reset_index(drop=True)
    m["close"] = m["close_ref"].where(m["close_ref"].notna(), m["close_binance"])
    m["source"] = np.where(m["close_ref"].notna(), reference_name, "binance")
    m = m.dropna(subset=["close"]).reset_index(drop=True)
    m["log_return"] = np.log(m["close"]).diff()
    return m[["date", "close", "log_return", "source"]]


def _empty_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
