"""Standardized-distribution helpers shared by the parametric models.

``arch`` fits a **standardized** Student-t (unit variance). The plain
``scipy.stats.t`` quantile has variance ``nu/(nu-2)``; rescale by
``sqrt((nu-2)/nu)`` to get the unit-variance quantile. (This is the bug that
made v1's GARCH VaR ~1.7x too wide.)
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def normal_z():
    """(ppf, cdf, es) for the standard normal lower tail."""
    ppf = stats.norm.ppf
    cdf = stats.norm.cdf
    es = lambda a: -stats.norm.pdf(stats.norm.ppf(a)) / a  # noqa: E731
    return ppf, cdf, es


def student_t_z(nu: float):
    """(ppf, cdf, es) for a UNIT-VARIANCE Student-t with ``nu`` d.o.f. (nu > 2)."""
    if nu <= 2.0:
        raise ValueError(f"student_t_z needs nu > 2, got {nu}")
    s = np.sqrt((nu - 2.0) / nu)

    def ppf(a: float) -> float:
        return float(stats.t.ppf(a, nu) * s)

    def cdf(x: float) -> float:
        return float(stats.t.cdf(x / s, nu))

    def es(a: float) -> float:
        q = stats.t.ppf(a, nu)  # raw-t quantile
        return float(-(stats.t.pdf(q, nu) / a) * ((nu + q**2) / (nu - 1.0)) * s)

    return ppf, cdf, es
