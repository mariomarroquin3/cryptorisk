"""Market-structure block: perp funding, open interest, exchange netflows,
stablecoin supply change (V2_PLAN §2, §3).

**Partial by design.** Free history is limited: Binance perp funding goes back
to the contract listing (~2019-09 for BTCUSDT); open-interest history is ~30
days; exchange netflows and stablecoin supply need a paid on-chain provider and
are left NULL. The ``-X`` models and the conditional predictive-ability analysis
degrade gracefully to "no exogenous block" when a column is missing
(V2_PLAN §10).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptorisk.data._http import get_json

_PERP = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}
_FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"
_OI_HIST = "https://fapi.binance.com/futures/data/openInterestHist"


def fetch_funding_daily(asset: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Columns: date, funding_8h  (mean of the day's 8h funding rates)."""
    sym = _PERP[asset]
    t0 = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    t1 = int((pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.utcnow()).timestamp() * 1000)
    rows, cursor = [], t0
    while cursor < t1:
        batch = get_json(_FUNDING, params={"symbol": sym, "startTime": cursor, "endTime": t1, "limit": 1000})
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1]["fundingTime"] + 1
        if len(batch) < 1000:
            break
    if not rows:
        return pd.DataFrame(columns=["date", "funding_8h"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(pd.to_numeric(df["fundingTime"]), unit="ms").dt.normalize()
    df["funding_8h"] = pd.to_numeric(df["fundingRate"])
    return df.groupby("date", as_index=False)["funding_8h"].mean()


def fetch_open_interest_daily(asset: str) -> pd.DataFrame:
    """Columns: date, open_interest. ~30 days of history only."""
    sym = _PERP[asset]
    try:
        js = get_json(_OI_HIST, params={"symbol": sym, "period": "1d", "limit": 500})
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=["date", "open_interest"])
    if not js:
        return pd.DataFrame(columns=["date", "open_interest"])
    df = pd.DataFrame(js)
    df["date"] = pd.to_datetime(pd.to_numeric(df["timestamp"]), unit="ms").dt.normalize()
    df["open_interest"] = pd.to_numeric(df["sumOpenInterest"])
    return df[["date", "open_interest"]]


def build_microstructure(asset: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Columns: date, funding_8h, open_interest, netflow, stbl_supply_chg
    (last two always NULL for now)."""
    cal = pd.DataFrame({"date": pd.date_range(start, end or pd.Timestamp.utcnow().normalize(), freq="D")})
    out = cal.merge(fetch_funding_daily(asset, start, end), on="date", how="left")
    out = out.merge(fetch_open_interest_daily(asset), on="date", how="left")
    out["netflow"] = np.nan
    out["stbl_supply_chg"] = np.nan
    return out[["date", "funding_8h", "open_interest", "netflow", "stbl_supply_chg"]]
