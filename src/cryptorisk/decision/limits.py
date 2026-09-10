"""Position-limit framework and its own backtest (V2_PLAN §6.2).

N* such that ES_{99%,1d}(N*) <= risk budget; utilisation series
ES_{99%,1d}(N_t)/budget; and a backtest of the framework itself - how often the
chosen limit would have been breached and whether breaches line up with large
realized losses.

Phase 5.
"""

from __future__ import annotations


def max_notional(es_99_1d_per_unit: float, risk_budget: float) -> float:
    raise NotImplementedError("decision.limits.max_notional - Phase 5")


def backtest_framework(es_series, realized_pnl, limit_notional: float):
    raise NotImplementedError("decision.limits.backtest_framework - Phase 5")
