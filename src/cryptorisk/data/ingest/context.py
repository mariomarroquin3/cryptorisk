"""Descriptive context: on-chain (hashrate, difficulty) + macro (SPX, DXY,
fed funds, CPI) -- V2_PLAN §2.

Never a feature of the daily VaR; kept for the dashboard / narrative only.
Hashrate & difficulty from blockchain.info; SPX & DXY from Yahoo. Fed funds and
CPI come from FRED, which is not reachable from every environment - those two
columns stay NULL when the fetch fails (they are optional context).
"""

from __future__ import annotations

import io

import pandas as pd

from cryptorisk.data._http import get_json

_BLOCKCHAIN = "https://api.blockchain.info/charts/{chart}"
_YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
_FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def _blockchain_chart(chart: str) -> pd.DataFrame:
    js = get_json(_BLOCKCHAIN.format(chart=chart), params={"timespan": "all", "format": "json", "sampled": "false"})
    v = js.get("values", [])
    df = pd.DataFrame(v)
    if df.empty:
        return pd.DataFrame(columns=["date", chart])
    df["date"] = pd.to_datetime(df["x"], unit="s").dt.normalize()
    return df.rename(columns={"y": chart})[["date", chart]]


def _yahoo_series(symbol: str, name: str, start: str, end: str | None) -> pd.DataFrame:
    p0 = int(pd.Timestamp(start, tz="UTC").timestamp())
    p1 = int((pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.utcnow()).timestamp())
    js = get_json(_YAHOO.format(sym=symbol), params={"period1": p0, "period2": p1, "interval": "1d"})
    res = js["chart"]["result"]
    if not res:
        return pd.DataFrame(columns=["date", name])
    r = res[0]
    ts = r.get("timestamp") or []
    close = r["indicators"]["quote"][0].get("close") or []
    df = pd.DataFrame({"date": pd.to_datetime(ts, unit="s").normalize(), name: close})
    return df.dropna().drop_duplicates("date")


def _fred_series(series_id: str, name: str) -> pd.DataFrame:
    try:
        import requests

        raw = requests.get(_FRED_CSV, params={"id": series_id}, timeout=30,
                           headers={"User-Agent": "cryptorisk/0.1"}).content
        df = pd.read_csv(io.BytesIO(raw))
        df.columns = ["date", name]
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df[name] = pd.to_numeric(df[name], errors="coerce")
        return df.dropna()
    except Exception:  # noqa: BLE001 - FRED is optional context
        return pd.DataFrame(columns=["date", name])


def build_context(start: str, end: str | None = None, *, cpi_publication_lag_days: int = 45) -> pd.DataFrame:
    """Columns: date, spx, dxy, fed_funds, cpi_lag, hashrate, difficulty
    on a daily calendar from ``start``; slow macro series forward-filled."""
    cal = pd.DataFrame({"date": pd.date_range(start, end or pd.Timestamp.utcnow().normalize(), freq="D")})

    parts = [
        _blockchain_chart("hash-rate").rename(columns={"hash-rate": "hashrate"}),
        _blockchain_chart("difficulty"),
        _yahoo_series("%5EGSPC", "spx", start, end),
        _yahoo_series("DX-Y.NYB", "dxy", start, end),
        _fred_series("DFF", "fed_funds"),
    ]

    cpi = _fred_series("CPIAUCSL", "cpi_lag")
    if not cpi.empty:
        cpi["date"] = cpi["date"] + pd.Timedelta(days=cpi_publication_lag_days)

    out = cal
    for p in [*parts, cpi]:
        out = out.merge(p, on="date", how="left")

    for col in ("spx", "dxy", "fed_funds", "cpi_lag", "hashrate", "difficulty"):
        if col in out.columns:
            out[col] = out[col].ffill()
        else:
            out[col] = pd.NA
    return out[["date", "spx", "dxy", "fed_funds", "cpi_lag", "hashrate", "difficulty"]]
