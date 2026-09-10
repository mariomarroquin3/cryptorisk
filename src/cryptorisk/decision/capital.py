"""Risk capital, FRTB Internal-Models style (V2_PLAN §6.1).

capital = m_c * ES_{97.5%, LH}, with a liquidity horizon LH >= 10d, an internal
multiplier m_c = base + Basel traffic-light add-on, and a model-risk add-on
equal to the capital spread between the best and worst model in the MCS
(analogous to an AVA / prudent-valuation adjustment).

Phase 5.
"""

from __future__ import annotations


def es_capital(es_975_1d: float, *, liquidity_horizon: int, multiplier: float, notional: float) -> float:
    raise NotImplementedError("decision.capital.es_capital - Phase 5")


def model_risk_addon(capital_by_model: dict[str, float], mcs_included: list[str]) -> float:
    raise NotImplementedError("decision.capital.model_risk_addon - Phase 5")
