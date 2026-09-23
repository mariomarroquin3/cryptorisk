"""Keep the API's price history current without the (offline) DuckDB store.

``data/results/price_history.parquet`` is a snapshot; the API extends it at
read time with the complete UTC days since the snapshot, pulled from Binance
5-minute klines. The same bars give the realized measures (RV, BV, semivariances,
jumps, RQ) for those days, so the intraday-based models (HAR-RV, HARQ,
Realized-GARCH/SV, GARCH-X, RF-QR) see the new days too, not only the
close-to-close ones.

Also exposes the *current, incomplete* UTC day (return so far, running low) so a
forecast can be checked against what has already happened today.

Every function degrades to "no extension" on network failure: the API then
serves the snapshot exactly as before.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

from cryptorisk.data.realized import realized_daily

# data-api.binance.vision serves the same public market data without the
# geo-restriction that api.binance.com applies to some cloud regions (Render's
# included -- that host answers HTTP 451 from there).
HOSTS = ("https://data-api.binance.vision", "https://api.binance.com", "https://api1.binance.com")
SYMBOL = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "BNB": "BNBUSDT"}
_BAR_MS = 300_000
_BARS_PER_DAY = 288
_MIN_BARS = 280            # a day with fewer 5-min bars is treated as incomplete
_MAX_GAP_DAYS = 45         # beyond this the snapshot is too old to patch at read time
_SPOT_COLS = ("rv", "bv", "rsv_pos", "rsv_neg", "jump", "rq")


def get_json(path: str, params: dict, *, timeout: float = 6.0):
    """GET ``path`` from the first host that answers; ``None`` if all fail."""
    for host in HOSTS:
        try:
            r = requests.get(host + path, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError):
            continue
    return None


def fetch_5m(asset: str, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame | None:
    """5-min bars with ``lo <= open_time < hi`` (naive UTC), paginated. ``None``
    on failure; an empty frame when the range simply has no bars."""
    sym = SYMBOL.get(asset)
    if sym is None:
        return None
    t = int(lo.timestamp() * 1000)
    stop = int(hi.timestamp() * 1000)
    rows: list[list] = []
    while t < stop:
        batch = get_json(
            "/api/v3/klines",
            {"symbol": sym, "interval": "5m", "startTime": t, "endTime": stop - 1, "limit": 1000},
        )
        if batch is None:
            return None
        if not batch:
            break
        rows += batch
        t = int(batch[-1][0]) + _BAR_MS
    if not rows:
        return pd.DataFrame(columns=["ts", "close"])
    df = pd.DataFrame(rows).iloc[:, [0, 4]]
    df.columns = ["open_time", "close"]
    out = pd.DataFrame(
        {"ts": pd.to_datetime(pd.to_numeric(df["open_time"]), unit="ms"), "close": pd.to_numeric(df["close"])}
    )
    return out.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def _utc_today() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()


def extend_history(asset: str, hist: pd.DataFrame, *, today: pd.Timestamp | None = None) -> pd.DataFrame:
    """``hist`` (one asset, columns date/close/log_return/rv/...) plus a row for
    every complete UTC day after its last date. Returns ``hist`` unchanged if
    nothing can be added."""
    if hist.empty:
        return hist
    today = _utc_today() if today is None else today
    last = pd.Timestamp(hist["date"].max()).normalize()
    if last >= today - pd.Timedelta(days=1) or (today - last).days > _MAX_GAP_DAYS:
        return hist

    # Start at the last bar of `last`'s day so the first new return (23:55 -> 00:00) exists.
    bars = fetch_5m(asset, last + pd.Timedelta(days=1) - pd.Timedelta(minutes=5), today)
    if bars is None or bars.empty:
        return hist

    day = bars["ts"].dt.normalize()
    new = bars[day > last]
    counts = new.groupby(new["ts"].dt.normalize())["close"].size()
    complete = counts[counts >= _MIN_BARS].index
    if len(complete) == 0:
        return hist
    closes = new[new["ts"].dt.normalize().isin(complete)].groupby(new["ts"].dt.normalize())["close"].last()

    prev = float(hist.sort_values("date")["close"].iloc[-1])
    rv = realized_daily(bars).assign(date=lambda d: pd.to_datetime(d["date"])).set_index("date")
    rows = []
    for d, c in closes.items():
        # A gap in the sequence (an incomplete day in the middle) would make the
        # return span two days; stop rather than record a misleading one.
        if d != last + pd.Timedelta(days=1) + pd.Timedelta(days=len(rows)):
            break
        row = {"date": d, "close": float(c), "log_return": float(np.log(c / prev))}
        for col in _SPOT_COLS:
            row[col] = float(rv.at[d, col]) if d in rv.index else np.nan
        rows.append(row)
        prev = float(c)
    if not rows:
        return hist
    add = pd.DataFrame(rows)
    if "asset" in hist:
        add["asset"] = asset
    return pd.concat([hist, add[hist.columns.intersection(add.columns)]], ignore_index=True)[list(hist.columns)]


def today_so_far(asset: str, ref_close: float, *, now: pd.Timestamp | None = None) -> dict | None:
    """Return and running low of the current, incomplete UTC day, measured from
    ``ref_close`` (the previous complete close). ``None`` if unavailable."""
    now = pd.Timestamp.now(tz="UTC").tz_localize(None) if now is None else now
    start = now.normalize()
    bars = fetch_5m(asset, start, now + pd.Timedelta(minutes=5))
    if bars is None or bars.empty:
        return None
    px = bars["close"].to_numpy(float)
    return {
        "date": str(start.date()),
        "n_bars": int(len(px)),
        "fraction_of_day": float(min(len(px) / _BARS_PER_DAY, 1.0)),
        "price": float(px[-1]),
        "ret_so_far": float(np.log(px[-1] / ref_close)),
        "low_ret": float(np.log(px.min() / ref_close)),
        "high_ret": float(np.log(px.max() / ref_close)),
        "fetched_at": time.time(),
    }
