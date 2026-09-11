"""Cached data access for the dashboard: study outputs, the DuckDB store, a
polled spot price, and an on-demand "today" forecast.

Everything here is read-only against ``data/results/`` and the store. Cache
TTLs are short (live price) to long (backtest outputs, which only change when
the pipeline re-runs) -- see each function.
"""

from __future__ import annotations

import time
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import requests
import streamlit as st

from cryptorisk.config import load_config, repo_root
from cryptorisk.models.base import Context
from cryptorisk.models.registry import all_models

_RESULTS = repo_root() / "data" / "results"

# Binance spot symbols for the assets the study covers.
_BINANCE_SYMBOL = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "BNB": "BNBUSDT"}

_REALIZED_COLS = ("rv", "bv", "rsv_pos", "rsv_neg", "jump", "rq")
_NEEDS_REALIZED = {"HAR-RV", "HARQ", "Realized-GARCH", "GARCH-X"}


def _csv(name: str) -> pd.DataFrame:
    path = _RESULTS / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=300, show_spinner=False)
def load_results() -> dict[str, pd.DataFrame]:
    """All the small `data/results/*.csv` tables the study writes."""
    return {
        "fz0_mcs": _csv("eval_fz0_mcs.csv"),
        "coverage": _csv("eval_coverage.csv"),
        "es": _csv("eval_es.csv"),
        "volforecast": _csv("eval_volforecast.csv"),
        "backtests_summary": _csv("backtests_summary.csv"),
        "capital": _csv("decision_capital.csv"),
        "estimation_risk": _csv("decision_estimation_risk.csv"),
        "limits": _csv("decision_limits.csv"),
        "hedge": _csv("decision_hedge.csv"),
        "pla": _csv("decision_pla.csv"),
        "portfolio_eval": _csv("portfolio_eval.csv"),
        "regime": _csv("regime_identification.csv"),
        "gw_cpa": _csv("eval_gw_cpa.csv"),
    }


@st.cache_data(ttl=300, show_spinner=False)
def load_backtests() -> pd.DataFrame:
    path = _RESULTS / "backtests.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def load_portfolio_backtests() -> pd.DataFrame:
    path = _RESULTS / "portfolio_backtests.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@st.cache_resource(show_spinner=False)
def store_path() -> str:
    cfg = load_config()
    return str(repo_root() / cfg["paths"]["store"])


@st.cache_data(ttl=300, show_spinner=False)
def load_price_window(asset: str, n: int = 900) -> pd.DataFrame:
    """Last ``n`` days of close/log-return, left-joined with realized measures."""
    con = duckdb.connect(store_path(), read_only=True)
    df = con.execute(
        """
        SELECT r.date, r.close, r.log_return,
               x.rv, x.bv, x.rsv_pos, x.rsv_neg, x.jump, x.rq
        FROM returns_daily r
        LEFT JOIN realized_daily x USING (asset, date)
        WHERE r.asset = ?
        ORDER BY r.date DESC
        LIMIT ?
        """,
        [asset, n],
    ).df()
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=15, show_spinner=False)
def live_price(asset: str) -> dict[str, float] | None:
    """Poll Binance's public 24h-ticker endpoint. No key required."""
    symbol = _BINANCE_SYMBOL.get(asset)
    if symbol is None:
        return None
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/ticker/24hr",
            params={"symbol": symbol},
            timeout=4,
        )
        r.raise_for_status()
        d = r.json()
        return {
            "price": float(d["lastPrice"]),
            "change_pct": float(d["priceChangePercent"]),
            "high": float(d["highPrice"]),
            "low": float(d["lowPrice"]),
            "volume": float(d["volume"]),
            "fetched_at": time.time(),
        }
    except Exception:
        return None


def primary_model(asset: str, alpha: float, results: dict[str, pd.DataFrame]) -> str | None:
    """The FZ0-best, in-MCS model for (asset, alpha) per the last `evaluate` run."""
    df = results.get("fz0_mcs", pd.DataFrame())
    if df.empty:
        return None
    sub = df[(df["asset"] == asset) & (df["alpha"] == alpha) & (df["is_best"])]
    if sub.empty:
        return None
    return str(sub.iloc[0]["model"])


@st.cache_resource(show_spinner=False)
def _model_map() -> dict[str, Any]:
    return {m.name: m for m in all_models()}


def model_names() -> list[str]:
    return sorted(_model_map().keys())


@st.cache_data(ttl=600, show_spinner=False)
def today_forecast(
    asset: str,
    model_name: str,
    alphas: tuple[float, ...] = (0.01, 0.025),
    window: int = 500,
) -> dict[str, Any] | None:
    """Re-fit ``model_name`` on the latest ``window`` obs for a live, one-step-
    ahead VaR/ES band -- *not* a value from the frozen study backtest.

    Returns ``None`` if the model isn't in the registry or the fit fails
    (caller should fall back to the last stored backtest row instead).
    """
    model = _model_map().get(model_name)
    if model is None:
        return None
    win = load_price_window(asset, n=window + 30)
    if len(win) < window:
        return None
    win = win.tail(window)

    realized = None
    if model_name in _NEEDS_REALIZED:
        realized = {c: win[c].to_numpy(float) for c in _REALIZED_COLS if c in win}

    ctx = Context(
        returns=win["log_return"].to_numpy(float),
        dates=win["date"].to_numpy("datetime64[D]"),
        asof=win["date"].to_numpy("datetime64[D]")[-1],
        asset=asset,
        realized=realized,
    )
    try:
        dist = model.fit_predict(ctx)
    except Exception:
        return None

    out: dict[str, Any] = {
        "model": model_name,
        "asof": win["date"].iloc[-1],
        "last_close": float(win["close"].iloc[-1]),
    }
    for a in alphas:
        try:
            out[f"var_{a}"] = float(dist.var(a))
        except Exception:
            out[f"var_{a}"] = np.nan
        try:
            out[f"es_{a}"] = float(dist.es(a))
        except Exception:
            out[f"es_{a}"] = np.nan
        try:
            out[f"upper_{a}"] = float(dist.ppf(1 - a))
        except NotImplementedError:
            out[f"upper_{a}"] = np.nan
    return out
