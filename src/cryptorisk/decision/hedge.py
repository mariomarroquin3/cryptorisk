"""Perpetual-futures hedge analysis (V2_PLAN §6.4).

Minimum-variance and ES-minimising hedge ratios; annualised funding cost of the
hedge; ES / capital hedged vs unhedged; spot-perp basis risk and its behaviour
in the stress sub-periods.

Phase 5.
"""

from __future__ import annotations


def min_variance_ratio(spot_ret, perp_ret) -> float:
    raise NotImplementedError("decision.hedge.min_variance_ratio - Phase 5")


def es_minimising_ratio(spot_ret, perp_ret, alpha: float) -> float:
    raise NotImplementedError("decision.hedge.es_minimising_ratio - Phase 5")
