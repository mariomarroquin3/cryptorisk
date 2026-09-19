"""5-minute bars from data.binance.vision (V2_PLAN §2).

Monthly zip archives, one CSV each, no header. Columns (Binance spec):
open_time, open, high, low, close, volume, close_time, quote_volume, trades,
taker_buy_base, taker_buy_quote, ignore.

``open_time`` is epoch **milliseconds** in older files and **microseconds** in
newer ones - detected by magnitude.

Yielded month by month so the caller streams into the store instead of holding
years of 5-min bars in memory.
"""

from __future__ import annotations

import io
import time
import zipfile
from collections.abc import Iterator

import pandas as pd
import requests

from cryptorisk.data._http import get_bytes, url_exists

_BASE = "https://data.binance.vision/data/spot/monthly/klines"
_SYMBOL = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "BNB": "BNBUSDT"}
_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "tbb", "tbq", "ignore",
]


def _months(start: str, end: str | None) -> list[pd.Timestamp]:
    last = (pd.Timestamp(end) if end else pd.Timestamp.now(tz="UTC").tz_localize(None)).normalize().replace(day=1)
    # only complete months
    last = last - pd.offsets.MonthBegin(1)
    return list(pd.date_range(pd.Timestamp(start).replace(day=1), last, freq="MS"))


def _parse_month(raw: bytes, symbol: str) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as fh:
            head = fh.read(64)
    has_header = b"open_time" in head.lower()
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        df = pd.read_csv(
            zf.open(zf.namelist()[0]),
            header=0 if has_header else None,
            names=None if has_header else _COLS,
        )
    df.columns = [str(c).strip() for c in df.columns]
    ot = pd.to_numeric(df["open_time"])
    unit = "us" if ot.iloc[0] > 10**14 else "ms"
    out = pd.DataFrame({
        "ts": pd.to_datetime(ot, unit=unit),
        "open": pd.to_numeric(df["open"]),
        "high": pd.to_numeric(df["high"]),
        "low": pd.to_numeric(df["low"]),
        "close": pd.to_numeric(df["close"]),
        "volume": pd.to_numeric(df["volume"]),
    })
    return out.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def iter_binance_5m(asset: str, start: str, end: str | None = None) -> Iterator[tuple[pd.Timestamp, pd.DataFrame]]:
    """Yield ``(month, bars)`` for each complete month in range. Missing months
    are skipped with no error."""
    sym = _SYMBOL[asset]
    for month in _months(start, end):
        stamp = f"{month.year}-{month.month:02d}"
        url = f"{_BASE}/{sym}/5m/{sym}-5m-{stamp}.zip"
        if not url_exists(url):
            continue
        yield month, _parse_month(get_bytes(url), sym)


_REST = "https://api.binance.com/api/v3/klines"
_BAR_MS = 300_000


def partial_month_range(start: str, end: str | None) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Half-open ``[first bar, end of last day)`` for the days the monthly
    archives cannot cover: data.binance.vision publishes a month only once it
    is over, so the in-progress month is missing. Runs from the first day after
    the last complete month through ``end`` inclusive (default: yesterday UTC,
    the last complete day). ``None`` when the range is empty."""
    last_day = (
        pd.Timestamp(end) if end else pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=1)
    ).normalize()
    months = _months(start, end)
    first = (months[-1] + pd.offsets.MonthBegin(1)) if months else pd.Timestamp(start).normalize()
    lo, hi = max(first, pd.Timestamp(start).normalize()), last_day + pd.Timedelta(days=1)
    return (lo, hi) if lo < hi else None


def fetch_rest_5m(asset: str, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame:
    """5-minute bars for ``[lo, hi)`` (naive UTC) from the REST klines
    endpoint, paginated at 1000 bars. Same schema as :func:`_parse_month`."""
    sym = _SYMBOL[asset]
    t = int(lo.timestamp() * 1000)
    stop = int(hi.timestamp() * 1000)
    rows: list[list] = []
    while t < stop:
        for attempt in range(4):
            try:
                r = requests.get(
                    _REST,
                    params={"symbol": sym, "interval": "5m", "startTime": t, "endTime": stop - 1, "limit": 1000},
                    timeout=20,
                )
                r.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
        batch = r.json()
        if not batch:
            break
        rows += batch
        t = int(batch[-1][0]) + _BAR_MS
    if not rows:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = ["open_time", "open", "high", "low", "close", "volume"]
    out = pd.DataFrame({
        "ts": pd.to_datetime(pd.to_numeric(df["open_time"]), unit="ms"),
        **{c: pd.to_numeric(df[c]) for c in ("open", "high", "low", "close", "volume")},
    })
    return out.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
