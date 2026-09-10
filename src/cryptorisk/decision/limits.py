"""Position-limit framework and its own backtest (V2_PLAN §6.2).

``N*`` is the largest notional whose 1-day 99% ES stays within the risk
budget.  The framework is then backtested on the OOS period:

* **utilisation** -- the series ``N* * |ES_{99,1d,t}| / budget``;
* **bind rate** -- how often the model's own risk estimate exceeds the budget
  (you would have had to cut the position);
* **budget breaches** -- how often the realized 1-day loss on the ``N*``
  position exceeds the budget, and whether the model flagged elevated risk the
  day before;
* **ES exceedances** -- realized loss above the day's risk estimate; should run
  at about ``alpha``.

A framework that never binds is as useless as one that always breaches.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def max_notional(es_99_1d_per_unit: float, risk_budget: float) -> float:
    """``N* = budget / |ES_per_unit|`` (ES as a negative return fraction)."""
    e = abs(float(es_99_1d_per_unit))
    return float("inf") if e == 0.0 else float(risk_budget) / e


@dataclass(frozen=True)
class LimitBacktest:
    limit_notional: float
    budget: float
    n: int
    bind_rate: float  # P(N* * |ES_t| > budget)
    mean_utilisation: float  # mean(N* * |ES_t| / budget)
    max_utilisation: float
    n_budget_breaches: int  # realized 1-day loss on N* > budget
    budget_breach_rate: float
    flagged_breach_rate: float  # of those, share where that day's ES already exceeded budget
    es_exceedance_rate: float  # realized loss > that day's ES estimate (below alpha: ES < VaR)
    worst_loss: float  # largest realized 1-day loss on the N* position


def backtest_framework(
    es_series,
    realized_returns,
    limit_notional: float,
    *,
    budget: float,
) -> LimitBacktest:
    es = np.abs(np.asarray(es_series, float))
    r = np.asarray(realized_returns, float)
    ok = np.isfinite(es) & np.isfinite(r)
    es, r = es[ok], r[ok]
    n = es.size

    risk_usd = limit_notional * es
    loss_usd = -r * limit_notional  # positive = a loss

    bind = risk_usd > budget
    breach = loss_usd > budget
    # a breach is "flagged" if the model's own day-ahead ES for that day already
    # exceeded the budget (es_t is the forecast made at t-1 for day t)
    flagged = breach & bind

    return LimitBacktest(
        limit_notional=float(limit_notional),
        budget=float(budget),
        n=int(n),
        bind_rate=float(bind.mean()) if n else np.nan,
        mean_utilisation=float((risk_usd / budget).mean()) if n else np.nan,
        max_utilisation=float((risk_usd / budget).max()) if n else np.nan,
        n_budget_breaches=int(breach.sum()),
        budget_breach_rate=float(breach.mean()) if n else np.nan,
        flagged_breach_rate=float(flagged.sum() / breach.sum()) if breach.any() else np.nan,
        es_exceedance_rate=float((loss_usd > risk_usd).mean()) if n else np.nan,
        worst_loss=float(loss_usd.max()) if n else np.nan,
    )
