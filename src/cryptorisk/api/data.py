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
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from cryptorisk.api import live_tail
from cryptorisk.api.cache import ttl_cache
from cryptorisk.config import repo_root
from cryptorisk.models.base import Context
from cryptorisk.models.registry import all_models

_RESULTS = repo_root() / "data" / "results"

_BINANCE_SYMBOL = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "BNB": "BNBUSDT"}
_REALIZED_COLS = ("rv", "bv", "rsv_pos", "rsv_neg", "jump", "rq")
_NEEDS_REALIZED = {"HAR-RV", "HARQ", "Realized-GARCH", "Realized-SV", "GARCH-X", "RF-QR"}


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
        "rf_importance": _csv("explain_rf_importance.csv"),
        "rf_diagnostics": _csv("explain_rf_diagnostics.csv"),
        "rf_inputs": _csv("explain_rf_inputs.csv"),
        "lstm_importance": _csv("explain_lstm_importance.csv"),
    }


@ttl_cache(300)
def load_backtests() -> pd.DataFrame:
    path = _RESULTS / "backtests.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@ttl_cache(300)
def latest_by_model(asset: str, alpha: float, window: int = 500) -> pd.DataFrame:
    """Every model's most recent frozen (date, VaR, ES) for one (asset, alpha)
    -- the whole model-risk spread on the same day, one row per model. Used
    for the Overview's model-agreement strip plot; a single call over the
    already-cached ``backtests.parquet`` instead of one `/backtests` request
    per model."""
    bt = load_backtests()
    if bt.empty:
        return bt
    sub = bt[(bt.asset == asset) & (bt.alpha == alpha) & (bt.window == window)]
    if sub.empty:
        return sub
    return (
        sub.sort_values("date")
        .groupby("model", as_index=False, observed=True)
        .tail(1)[["model", "date", "var", "es", "violation"]]
        .reset_index(drop=True)
    )


@lru_cache(maxsize=1)
def _snapshot_history() -> pd.DataFrame:
    path = _RESULTS / "price_history.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


@ttl_cache(600)
def _price_history() -> pd.DataFrame:
    """The committed snapshot plus every complete UTC day since it, from Binance
    5-min klines (``api.live_tail``): the forecast and price chart stay current
    without a redeploy. Falls back to the bare snapshot if Binance is unreachable."""
    snap = _snapshot_history()
    if snap.empty or "asset" not in snap:
        return snap
    groups = list(snap.groupby("asset", sort=False))
    with ThreadPoolExecutor(max_workers=len(groups)) as pool:
        parts = list(pool.map(lambda ag: live_tail.extend_history(ag[0], ag[1]), groups))
    return pd.concat(parts, ignore_index=True)


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
    d = live_tail.get_json("/api/v3/ticker/24hr", {"symbol": symbol}, timeout=4)
    if d is None:
        return None
    try:
        return {
            "price": float(d["lastPrice"]),
            "change_pct": float(d["priceChangePercent"]),
            "high": float(d["highPrice"]),
            "low": float(d["lowPrice"]),
            "volume": float(d["volume"]),
            "fetched_at": time.time(),
        }
    except (KeyError, TypeError, ValueError):
        return None


@ttl_cache(60)
def intraday_status(asset: str, ref_close: float) -> dict[str, Any] | None:
    """Today's incomplete UTC day against the previous complete close."""
    return live_tail.today_so_far(asset, ref_close)


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


@ttl_cache(300)
def regime_stats(asset: str) -> list[dict[str, Any]]:
    """Return moments by regime, for the two ways the regime is labelled: the
    full-sample fit (in-sample, sees the future) and the walk-forward one-step
    P(crisis) (what a forecaster could act on). A day is "crisis" when P > 0.5.
    ``vol_ratio`` is sd(crisis)/sd(normal) and ``levene_p`` tests equal
    variances: if a labelling separates real volatility states, the ratio is
    well above 1 and the test rejects."""
    from scipy import stats

    path = _RESULTS / f"msgarch_pred_{asset}.csv"
    ph = _price_history()
    if not path.exists() or ph.empty:
        return []
    pred = pd.read_csv(path, usecols=["date", "prob_crisis_pred", "prob_crisis_insample"])
    pred["date"] = pd.to_datetime(pred["date"])
    ret = ph[ph["asset"] == asset][["date", "log_return"]] if "asset" in ph else ph[["date", "log_return"]]
    df = pred.merge(ret, on="date", how="inner").dropna()
    out: list[dict[str, Any]] = []
    for source, col in (("in-sample fit", "prob_crisis_insample"), ("walk-forward", "prob_crisis_pred")):
        crisis = df[col] > 0.5
        groups = {"Normal": df.loc[~crisis, "log_return"].to_numpy(), "Crisis": df.loc[crisis, "log_return"].to_numpy()}
        sd = {k: float(np.std(v, ddof=1)) if v.size > 2 else float("nan") for k, v in groups.items()}
        ratio = sd["Crisis"] / sd["Normal"] if sd["Normal"] else float("nan")
        p = float("nan")
        if groups["Normal"].size > 2 and groups["Crisis"].size > 2:
            p = float(stats.levene(groups["Normal"], groups["Crisis"]).pvalue)
        for regime, x in groups.items():
            n = int(x.size)
            out.append({
                "source": source,
                "regime": regime,
                "n": n,
                "share": n / len(df) if len(df) else float("nan"),
                "mean": float(np.mean(x)) if n else float("nan"),
                "sd": sd[regime],
                "skew": float(stats.skew(x)) if n > 2 else float("nan"),
                "kurt": float(stats.kurtosis(x)) if n > 3 else float("nan"),
                "mean_abs": float(np.mean(np.abs(x))) if n else float("nan"),
                "vol_ratio": ratio,
                "levene_p": p,
            })
    return out


@ttl_cache(300)
def pcrisis_band(asset: str) -> pd.DataFrame:
    """Bootstrap parameter-uncertainty band for the walk-forward P(crisis)
    (``msgarch/bootstrap_pcrisis.R``); empty when the study was not run."""
    frames = [pd.read_csv(f) for f in sorted(_RESULTS.glob(f"msgarch_pcrisis_band_{asset}.csv"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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


@ttl_cache(300)
def rf_partial_dependence(
    asset: str, feature: str, alpha: float, window: int = 500
) -> dict[str, Any] | None:
    """RF-QR's partial dependence for one feature on the latest live window --
    same window ``today_forecast`` re-fits on. ``None`` if the window is too
    short, RF-QR isn't registered, or ``feature`` doesn't apply to this asset
    (e.g. a log-RV feature when realized measures are absent)."""
    from cryptorisk.models.random_forest import RandomForestQR

    model = _model_map().get("RF-QR")
    if not isinstance(model, RandomForestQR):
        return None
    win = load_price_window(asset, n=window + 30)
    if len(win) < window:
        return None
    win = win.tail(window)
    realized = {c: win[c].to_numpy(float) for c in _REALIZED_COLS if c in win}
    ctx = Context(
        returns=win["log_return"].to_numpy(float),
        dates=win["date"].to_numpy("datetime64[D]"),
        asof=win["date"].to_numpy("datetime64[D]")[-1],
        asset=asset,
        realized=realized,
    )
    try:
        return model.partial_dependence(ctx, feature, alpha=alpha)
    except Exception:  # noqa: BLE001 - malformed feature name etc. -> caller shows "unavailable"
        return None


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
