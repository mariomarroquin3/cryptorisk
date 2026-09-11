"""Read-only data access for the API: study outputs, the DuckDB store, a
polled spot price, and an on-demand "today" forecast.

Deliberately independent of ``cryptorisk.dashboard.data`` (no Streamlit
import) even though the logic mirrors it closely -- see the package
docstring. Keep the two in sync by hand if the underlying study schema
changes; they're small and simple enough that this is cheaper than forcing a
shared, framework-agnostic layer through two very different caching models.
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import requests

from cryptorisk.api.cache import ttl_cache
from cryptorisk.config import load_config, repo_root
from cryptorisk.models.base import Context
from cryptorisk.models.registry import all_models

_RESULTS = repo_root() / "data" / "results"

_BINANCE_SYMBOL = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "BNB": "BNBUSDT"}
_REALIZED_COLS = ("rv", "bv", "rsv_pos", "rsv_neg", "jump", "rq")
_NEEDS_REALIZED = {"HAR-RV", "HARQ", "Realized-GARCH", "GARCH-X"}


def _csv(name: str) -> pd.DataFrame:
    path = _RESULTS / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@ttl_cache(300)
def load_results() -> dict[str, pd.DataFrame]:
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


@ttl_cache(300)
def load_backtests() -> pd.DataFrame:
    path = _RESULTS / "backtests.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@lru_cache(maxsize=1)
def store_path() -> str:
    cfg = load_config()
    return str(repo_root() / cfg["paths"]["store"])


@ttl_cache(300)
def load_price_window(asset: str, n: int = 900) -> pd.DataFrame:
    con = duckdb.connect(store_path(), read_only=True)
    try:
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
    finally:
        con.close()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


@ttl_cache(15)
def live_price(asset: str) -> dict[str, float] | None:
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


@ttl_cache(300)
def regime_series(asset: str, limit: int = 2000) -> pd.DataFrame:
    con = duckdb.connect(store_path(), read_only=True)
    try:
        exists = con.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name='msgarch_predictions'"
        ).fetchone()[0]
        if not exists:
            return pd.DataFrame()
        df = con.execute(
            "SELECT date, sigma2, prob_crisis_insample, prob_crisis_pred "
            "FROM msgarch_predictions WHERE asset = ? ORDER BY date DESC LIMIT ?",
            [asset, limit],
        ).df()
    finally:
        con.close()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def primary_model(asset: str, alpha: float, results: dict[str, pd.DataFrame]) -> str | None:
    df = results.get("fz0_mcs", pd.DataFrame())
    if df.empty:
        return None
    sub = df[(df["asset"] == asset) & (df["alpha"] == alpha) & (df["is_best"])]
    return None if sub.empty else str(sub.iloc[0]["model"])


@lru_cache(maxsize=1)
def _model_map() -> dict[str, Any]:
    return {m.name: m for m in all_models()}


def model_names() -> list[str]:
    return sorted(_model_map().keys())


@ttl_cache(600)
def today_forecast(
    asset: str,
    model_name: str,
    alphas: tuple[float, ...] = (0.01, 0.025),
    window: int = 500,
) -> dict[str, Any] | None:
    """Re-fit ``model_name`` on the latest ``window`` obs for a live, one-step-
    ahead VaR/ES band -- *not* a value from the frozen study backtest.

    Returns ``None`` if the model isn't registered or the fit fails; callers
    should fall back to the last stored backtest row.
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
