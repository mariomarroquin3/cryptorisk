"""P&L attribution and the FRTB PLA test (V2_PLAN §6.3).

Decompose simulated daily P&L into drift / diffusion / jump / residual, then
compare model P&L (RTPL) with realized P&L (HPL) via Spearman correlation and
the Kolmogorov-Smirnov statistic -> green / amber / red.

Phase 5.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PLAResult:
    spearman: float
    ks: float
    zone: str


def pla_test(rtpl, hpl) -> PLAResult:
    raise NotImplementedError("decision.pnl_attribution.pla_test - Phase 5")
