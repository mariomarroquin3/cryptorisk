"""Read-only data access for the API: `data/results/` snapshots, a polled
spot price, and an on-demand "today" forecast.

Deliberately independent of ``cryptorisk.dashboard.data`` (no Streamlit
import, and no DuckDB store dependency -- the dashboard reads the store
directly, the API reads only `data/results/`, see `export_price_history`)
even though the logic mirrors it closely. Keep the two in sync by hand if
the underlying study schema changes; they're small and simple enough that
this is cheaper than forcing a shared, framework-agnostic layer through two
very different caching models.
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import requests

from cryptorisk.api.cache import ttl_cache
from cryptorisk.config import repo_root
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
def _price_history() -> pd.DataFrame:
    path = _RESULTS / "price_history.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


@ttl_cache(300)
def load_price_window(asset: str, n: int = 900) -> pd.DataFrame:
    df = _price_history()
    if df.empty:
        return df
    sub = df[df["asset"] == asset].sort_values("date")
    cols = ["date", "close", "log_return", "rv", "bv", "rsv_pos", "rsv_neg", "jump", "rq"]
    return sub.tail(n)[cols].reset_index(drop=True)


@ttl_cache(600)
def window_returns(asset: str, window: int = 500) -> np.ndarray | None:
    """The same estimation window ``today_forecast`` re-fits on, as a plain
    log-return array -- for ``api.cone``'s model-agnostic multi-day cone,
    which re-fits its own (Jump-Diffusion, GARCH-EVT) models independent of
    whichever model the `/forecast` `model` query param selected."""
    win = load_price_window(asset, n=window + 30)
    if len(win) < window:
        return None
    return win.tail(window)["log_return"].to_numpy(float)


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
    path = _RESULTS / f"msgarch_pred_{asset}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    cols = ["date", "sigma2", "prob_crisis_insample", "prob_crisis_pred"]
    return df.sort_values("date").tail(limit)[cols].reset_index(drop=True)


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
