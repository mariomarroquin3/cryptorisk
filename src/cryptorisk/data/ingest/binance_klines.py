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
import zipfile
from collections.abc import Iterator

import pandas as pd

from cryptorisk.data._http import get_bytes, url_exists

_BASE = "https://data.binance.vision/data/spot/monthly/klines"
_SYMBOL = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}
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
