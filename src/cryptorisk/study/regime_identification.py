"""Identification vs. estimation-window study for the 2-regime MS-GARCH
(V2_PLAN §5.5).

For W in {500, 750, 1000, 1500, 2000, expanding}: corr(P(high-vol)_filtered,
|r|) and corr with monthly RV, in-sample vs walk-forward; constrained specs
(regimes share alpha, beta) and FitMCMC. Output: "how much data does the
2-regime model need to separate the states".

Reuses the v1 scaffolding (``regime_window_exp.R``, ``constrained_regime.R``).
Phase 4.
"""

from __future__ import annotations


def run() -> None:
    raise NotImplementedError("study.regime_identification - Phase 4")
