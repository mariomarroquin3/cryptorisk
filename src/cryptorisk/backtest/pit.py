"""Probability Integral Transform diagnostics (V2_PLAN §5.6).

u_t = F_t(r_t); under a correct model u_t ~ iid U(0, 1). Berkowitz (2001) LR
test on Phi^{-1}(u_t) for normality + independence.

Phase 3.
"""

from __future__ import annotations

import numpy as np


def pit_values(cdf_at_realized: np.ndarray) -> np.ndarray:
    raise NotImplementedError("pit.pit_values - Phase 3")


def berkowitz(pit: np.ndarray) -> tuple[float, float]:
    raise NotImplementedError("pit.berkowitz - Phase 3")
