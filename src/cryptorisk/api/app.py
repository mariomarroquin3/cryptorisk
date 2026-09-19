"""FastAPI app: read-only REST endpoints over the study's pipeline outputs.

``make api`` (or ``uvicorn cryptorisk.api.app:app --reload``), then
``/docs`` for interactive Swagger docs. No auth -- read-only personal
project, deployed for a single known frontend origin (see ``CORS_ORIGINS``
below), not a multi-tenant service.

Every endpoint reads already-computed `data/results/` output; the one
exception is `/forecast/{asset}`, which re-fits the chosen model in-process
for a live, one-step-ahead band (`source: "live_refit"` in the response) and
falls back to the last frozen backtest row (`source: "frozen_backtest"`) if
that re-fit isn't supported. See `cryptorisk.api.data.today_forecast`.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from cryptorisk.api import cone as C
from cryptorisk.api import data as D
from cryptorisk.api.cache import ttl_cache
from cryptorisk.backtest.coverage import basel_zone_and_addon
from cryptorisk.config import load_config
from cryptorisk.models.notes import COPULA_NOTES as _COPULA_NOTES
from cryptorisk.models.notes import METHOD_FAMILY as _METHOD_FAMILY
from cryptorisk.models.notes import MODEL_NOTES as _MODEL_NOTES
from cryptorisk.study.run_portfolio import _read_bullets

app = FastAPI(
    title="CUBO+ Risk API",
    description=(
        "Read-only REST API over the cryptorisk study: VaR/ES per model, "
        "FZ0/MCS ranking, coverage & ES tests, the 4-asset portfolio, FRTB "
        "capital, position limits, the perp hedge, MS-GARCH regimes, plus a "
        "live spot price and an on-demand model re-fit."
    ),
    version="1.0.0",
)

# CORS_ORIGINS: comma-separated allowed origins (e.g. the deployed Vercel
# URL). Defaults to the two local dev ports so `make dev` / `make web` keep
# working out of the box; set explicitly in production.
_cors_env = os.environ.get("CORS_ORIGINS", "").strip()
_cors_origins = (
    [o.strip() for o in _cors_env.split(",") if o.strip()]
    if _cors_env
    else ["http://localhost:3000", "http://127.0.0.1:3000"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """DataFrame -> JSON-safe records (handles NaN/NaT, numpy scalars, dates)."""
    if df.empty:
        return []
    return json.loads(df.to_json(orient="records", date_format="iso"))


def _price(last_close: float, log_return: float | None) -> float | None:
    if log_return is None or not math.isfinite(log_return):
        return None
    return last_close * math.exp(log_return)


_CONE_HORIZONS_DAYS = tuple(range(1, 31))


def _dist_row(last: float, days: int, dist, alpha: float) -> dict:
    if dist is None:
        return {"days": days, "var_price": None, "es_price": None, "upper_price": None}
    try:
        var, es, upper = dist.var(alpha), dist.es(alpha), dist.ppf(1 - alpha)
    except Exception:  # noqa: BLE001
        return {"days": days, "var_price": None, "es_price": None, "upper_price": None}
    return {
        "days": days,
        "var_price": _price(last, var),
        "es_price": _price(last, es),
        "upper_price": _price(last, upper),
    }


@ttl_cache(600)
def _ensemble_cone(asset: str, last: float, alpha: float) -> dict:
    """The two-model forward cone: Jump-Diffusion and GARCH-EVT re-fit live
    on the current window and extended to each horizon (see ``api.cone``).
    Deliberately independent of the `model` query param. MS-GARCH's regime
    info is *not* folded into this time-indexed cone -- see
    ``regime_summary`` (a separate, non-horizon distribution comparison) and
    ``api.cone``'s docstring for why."""
    horizons = list(_CONE_HORIZONS_DAYS)
    returns = D.window_returns(asset)
    if returns is None:
        empty = [{"days": h, "var_price": None, "es_price": None, "upper_price": None} for h in horizons]
        return {"horizons_days": horizons, "jump_diffusion": empty, "garch_evt": empty}

    jd = C.jump_diffusion_cone(returns, horizons)
    evt = C.garch_evt_cone(returns, horizons)
    return {
        "horizons_days": horizons,
        "jump_diffusion": [_dist_row(last, h, jd.get(h), alpha) for h in horizons],
        "garch_evt": [_dist_row(last, h, evt.get(h), alpha) for h in horizons],
    }


def _check_asset(asset: str, assets: list[str]) -> None:
    if asset not in assets:
        raise HTTPException(404, f"unknown asset {asset!r}; must be one of {assets}")


def _check_alpha(alpha: float, alphas: list[float]) -> None:
    if not any(math.isclose(alpha, a) for a in alphas):
        raise HTTPException(400, f"unknown alpha {alpha!r}; must be one of {alphas}")


@app.get("/")
def root() -> dict:
    return {"name": "CUBO+ Risk API", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/config")
def config() -> dict:
    cfg = load_config()
    return {
        "assets": cfg["assets"],
        "alphas": cfg["alphas"],
        "oos_start": str(cfg["sample"]["oos_start"]),
        "portfolio": cfg.get("portfolio", {}),
    }


@app.get("/models")
def models() -> list[str]:
    return D.model_names()


@app.get("/models/info")
def models_info() -> dict[str, dict]:
    """One-line idea, family, and known limitations per model -- the same
    text ``docs/model_cards/`` renders, for a UI to show inline (e.g. a
    tooltip on a model name) instead of leaving it as an unexplained
    string in a table. Also covers the portfolio study's model names:
    ``Direct-<model>`` (the registry model run on the basket return
    series -- same spec, so it reuses that model's own note) and
    ``Copula-<family>`` (a dependence-structure choice, not a registry
    model, from ``_COPULA_NOTES``)."""
    out = {
        name: {"family": _METHOD_FAMILY.get(name, ""), "idea": idea, "limitations": lims}
        for name, (idea, lims) in _MODEL_NOTES.items()
    }
    for name, (idea, lims) in _MODEL_NOTES.items():
        out[f"Direct-{name}"] = {
            "family": _METHOD_FAMILY.get(name, ""),
            "idea": f"{idea} Applied directly to the basket's own return series.",
            "limitations": lims,
        }
    for name, (idea, lims) in _COPULA_NOTES.items():
        out[name] = {"family": "Copula (dependence structure)", "idea": idea, "limitations": lims}
    return out


@app.get("/price/{asset}")
def price(asset: str) -> dict:
    p = D.live_price(asset)
    if p is None:
        raise HTTPException(503, f"live price unavailable for {asset!r}")
    return {"asset": asset, **p}


@app.get("/prices/{asset}")
def prices(asset: str, limit: int = Query(365, le=5000)) -> list[dict]:
    """Historical daily close/log-return -- for a frontend to draw its own
    price chart, since only the dashboard (Streamlit) can read the store
    directly. Not the study's estimation window; just close+return history."""
    cfg = load_config()
    _check_asset(asset, cfg["assets"])
    win = D.load_price_window(asset, n=limit)
    if win.empty:
        return []
    return _records(win[["date", "close", "log_return"]])


@app.get("/forecast/{asset}")
def forecast(
    asset: str,
    model: str | None = Query(
        None, description="Model name; defaults to the FZ0-best, in-MCS model for `alpha`"
    ),
    alpha: float = Query(0.025, description="Tail probability, e.g. 0.025 -> 97.5% VaR"),
) -> dict:
    cfg = load_config()
    _check_asset(asset, cfg["assets"])
    _check_alpha(alpha, cfg["alphas"])
    results = D.load_results()
    model_name = model or D.primary_model(asset, alpha, results)
    if model_name is None:
        raise HTTPException(
            404, "no FZ0-best model found for this (asset, alpha); pass `model` explicitly"
        )

    fc = D.today_forecast(asset, model_name, alphas=tuple(cfg["alphas"]))
    if fc is not None:
        lo, es, hi = fc.get(f"var_{alpha}"), fc.get(f"es_{alpha}"), fc.get(f"upper_{alpha}")
        last = fc["last_close"]
        return {
            "asset": asset,
            "model": model_name,
            "alpha": alpha,
            "source": "live_refit",
            "asof": str(fc["asof"]),
            "last_close": last,
            "var": lo,
            "es": es,
            "upper": hi,
            "var_price": _price(last, lo),
            "es_price": _price(last, es),
            "upper_price": _price(last, hi),
            "cone": _ensemble_cone(asset, last, alpha),
            "regime_summary": C.regime_summary(asset),
        }

    bt = D.load_backtests()
    row = bt[(bt.asset == asset) & (bt.model == model_name) & (bt.alpha == alpha)].tail(1)
    if row.empty:
        raise HTTPException(404, f"no forecast available for {asset}/{model_name}")
    r = row.iloc[0]
    return {
        "asset": asset,
        "model": model_name,
        "alpha": alpha,
        "source": "frozen_backtest",
        "date": str(r["date"]),
        "var": float(r["var"]),
        "es": float(r["es"]),
    }


@app.get("/models/comparison")
def models_comparison(asset: str, alpha: float) -> list[dict]:
    cfg = load_config()
    _check_asset(asset, cfg["assets"])
    _check_alpha(alpha, cfg["alphas"])
    df = D.load_results()["fz0_mcs"]
    if df.empty:
        return []
    sub = df[(df.asset == asset) & (df.alpha == alpha)].sort_values("fz0_rank")
    return _records(sub)


@app.get("/coverage")
def coverage(asset: str, alpha: float) -> list[dict]:
    cfg = load_config()
    _check_asset(asset, cfg["assets"])
    _check_alpha(alpha, cfg["alphas"])
    df = D.load_results()["coverage"]
    if df.empty:
        return []
    return _records(df[(df.asset == asset) & (df.alpha == alpha)].sort_values("model"))


@app.get("/es-tests")
def es_tests(asset: str, alpha: float) -> list[dict]:
    cfg = load_config()
    _check_asset(asset, cfg["assets"])
    _check_alpha(alpha, cfg["alphas"])
    df = D.load_results()["es"]
    if df.empty:
        return []
    return _records(df[(df.asset == asset) & (df.alpha == alpha)].sort_values("model"))


@app.get("/gw-cpa")
def gw_cpa(asset: str, alpha: float | None = None) -> list[dict]:
    df = D.load_results()["gw_cpa"]
    if df.empty:
        return []
    sub = df[df.asset == asset]
    if alpha is not None and "alpha" in df.columns:
        sub = sub[sub.alpha == alpha]
    return _records(sub)


@app.get("/backtests")
def backtests(
    asset: str,
    model: str,
    alpha: float,
    limit: int = Query(500, le=5000, description="Most recent N days"),
) -> list[dict]:
    bt = D.load_backtests()
    if bt.empty:
        return []
    sub = bt[(bt.asset == asset) & (bt.model == model) & (bt.alpha == alpha)]
    sub = sub.sort_values("date").tail(limit)
    return _records(sub)


@app.get("/explain/rf")
def explain_rf(asset: str) -> dict:
    """RF-QR explainability: per-feature importance and effective-sample-size
    diagnostics over time (refit every 20 OOS days), plus the latest forecast
    row's inputs as z-scores of their window (`study.run_explain`)."""
    _check_asset(asset, load_config()["assets"])
    R = D.load_results()
    out: dict[str, list[dict]] = {}
    for key, name in (("importance", "rf_importance"), ("diagnostics", "rf_diagnostics"), ("inputs", "rf_inputs")):
        df = R[name]
        out[key] = _records(df[df.asset == asset]) if not df.empty else []
    return out


@app.get("/portfolio/eval")
def portfolio_eval(alpha: float) -> list[dict]:
    df = D.load_results()["portfolio_eval"]
    if df.empty:
        return []
    sub = df[(df.asset == "PORTFOLIO") & (df.alpha == alpha)].sort_values("fz0_rank")
    return _records(sub)


@app.get("/portfolio/narrative")
def portfolio_narrative(alpha: float) -> list[str]:
    """Reuses the study's own narrative generator (`run_portfolio._read_bullets`)
    against the already-computed `portfolio_eval.csv` -- same prose that ships
    in docs/portfolio.md, scoped to one alpha and served live. Cheap: no
    model refit, just formatting over rows already loaded for /portfolio/eval."""
    df = D.load_results()["portfolio_eval"]
    if df.empty:
        return []
    sub = df[(df.asset == "PORTFOLIO") & (df.alpha == alpha)]
    if sub.empty:
        return []
    pcfg = load_config().get("portfolio", {})
    if not pcfg.get("assets"):
        return []
    return _read_bullets(sub, pcfg)


@app.get("/portfolio/composition")
def portfolio_composition() -> dict:
    cfg = load_config()
    port = cfg.get("portfolio", {})
    basket = port.get("assets", [])
    return {
        "weights": port.get("weights", {}),
        "prices": {a: D.live_price(a) for a in basket},
    }


@app.get("/capital")
def capital(asset: str) -> list[dict]:
    df = D.load_results()["capital"]
    if df.empty:
        return []
    out = df[df.asset == asset].sort_values("capital_usd").copy()
    out["basel_zone"] = out["exceptions_250d"].apply(lambda x: basel_zone_and_addon(int(x))[0])
    return _records(out)


@app.get("/estimation-risk")
def estimation_risk(asset: str) -> list[dict]:
    df = D.load_results()["estimation_risk"]
    if df.empty:
        return []
    return _records(df[df.asset == asset])


@app.get("/limits")
def limits(asset: str) -> list[dict]:
    df = D.load_results()["limits"]
    if df.empty:
        return []
    return _records(df[df.asset == asset])


@app.get("/hedge")
def hedge(asset: str) -> list[dict]:
    df = D.load_results()["hedge"]
    if df.empty:
        return []
    return _records(df[df.asset == asset])


@app.get("/regimes/{asset}")
def regimes(asset: str, limit: int = Query(2000, le=5000)) -> dict:
    cfg = load_config()
    _check_asset(asset, cfg["assets"])
    series = D.regime_series(asset, limit=limit)
    corr = D.load_results()["regime"]
    corr = corr[corr.asset == asset] if not corr.empty else corr
    return {
        "series": _records(series),
        "correlations": _records(corr),
        "regime_summary": C.regime_summary(asset),
    }
