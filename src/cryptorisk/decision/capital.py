"""Risk capital, FRTB Internal-Models style (V2_PLAN §6.1).

``capital = m_c * N * |ES_{97.5%, LH}|`` with

* a **liquidity horizon** ``LH >= 10`` business days -- the 1-day ES is carried
  to LH by square-root-of-time (``ES_LH = ES_1d * sqrt(LH)``) and, separately,
  by a stationary block bootstrap of LH-day compounded returns, so the two can
  be compared;
* an **internal multiplier** ``m_c = base + Basel traffic-light add-on`` (the
  add-on comes from the 250-day exception count, ``backtest.coverage``);
* a **model-risk add-on** = the capital spread between the best and worst model
  still in the Model Confidence Set -- analogous to an AVA / prudent-valuation
  adjustment (EU Delegated Reg. 2016/101).

All ES inputs are in the project sign convention: a negative return level.
"""

from __future__ import annotations

import numpy as np

from cryptorisk.backtest.coverage import basel_zone_and_addon


def es_horizon_sqrt_time(es_1d: float, horizon_days: int) -> float:
    """Square-root-of-time scaling of a 1-day ES to ``horizon_days``."""
    return float(es_1d) * float(np.sqrt(horizon_days))


def es_horizon_bootstrap(
    returns_1d,
    alpha: float,
    horizon_days: int,
    *,
    n_boot: int = 20_000,
    block_len: int = 10,
    seed: int | None = None,
) -> float:
    """Empirical ``horizon_days``-day ES from a stationary block bootstrap of the
    1-day return series. Returns a negative return level (same convention as the
    square-root-of-time estimate)."""
    r = np.asarray(returns_1d, float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < horizon_days + 5:
        return np.nan
    rng = np.random.default_rng(seed)
    p = 1.0 / max(block_len, 1.0)
    agg = np.empty(n_boot)
    for b in range(n_boot):
        cur = rng.integers(n)
        s = 0.0
        for _ in range(horizon_days):
            if rng.random() < p:
                cur = rng.integers(n)
            s += r[cur]
            cur = (cur + 1) % n
        agg[b] = s
    q = np.quantile(agg, alpha)
    tail = agg[agg <= q]
    return float(tail.mean()) if tail.size else float(q)


def es_capital(
    es_975_1d: float,
    *,
    liquidity_horizon: int,
    multiplier: float,
    notional: float,
) -> float:
    """Capital in currency units: ``m_c * N * |ES_1d| * sqrt(LH)``."""
    es_lh = es_horizon_sqrt_time(es_975_1d, liquidity_horizon)
    return float(multiplier) * float(notional) * abs(es_lh)


def basel_multiplier(exceptions_250d: int, *, base: float = 1.5) -> float:
    """``base`` + the Basel traffic-light capital add-on for the exception
    count over the last 250 days (green 0-4 -> +0, amber 5-9 -> +0.40..0.85,
    red >=10 -> +1.0). Delegates to :func:`cryptorisk.backtest.coverage.
    basel_zone_and_addon` -- the single source of truth for the add-on table."""
    _, addon = basel_zone_and_addon(exceptions_250d)
    return float(base) + addon


def model_risk_addon(capital_by_model: dict[str, float], mcs_included: list[str]) -> float:
    """Capital spread across the models still in the MCS: ``max - min``.

    Zero if fewer than two admissible models. A prudent add-on for the fact
    that the "best" model is not uniquely identified.
    """
    caps = [capital_by_model[m] for m in mcs_included if m in capital_by_model]
    caps = [c for c in caps if np.isfinite(c)]
    if len(caps) < 2:
        return 0.0
    return float(max(caps) - min(caps))
