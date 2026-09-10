"""Expected Shortfall backtests (V2_PLAN §5.2).

Acerbi & Szekely (2014), "Back-testing Expected Shortfall". Return / VaR / ES are
in the project's sign convention: ``var`` and ``es`` are (negative) return
levels, a violation on day t is ``r_t < var_t``.

* ``z1`` — tests ES *given* the VaR frequency is right: on violation days, is the
  average tail loss equal to ES?  E[Z1] = 0 under H0.
* ``z2`` — the recommended joint test of VaR frequency *and* ES magnitude.
  E[Z2] = 0 under H0; Z2 < 0 means realized tail losses are worse than the ES
  forecast (ES too optimistic).

Neither statistic has a clean closed-form null. ``pvalue_by_simulation`` draws
the null distribution from simulated return paths (the study layer builds these
from each model's predictive distribution).
"""

from __future__ import annotations

import numpy as np


def _tail_ratio_sum(r: np.ndarray, v: np.ndarray, e: np.ndarray) -> tuple[float, int]:
    hit = r < v
    if not hit.any():
        return 0.0, 0
    return float(-np.sum(r[hit] / e[hit])), int(hit.sum())


def z1(realized, var, es) -> float:
    r, v, e = (np.asarray(x, float) for x in (realized, var, es))
    s, n_hit = _tail_ratio_sum(r, v, e)   # s = -sum(r_t / e_t) over breaches
    if n_hit == 0:
        return np.nan
    return s / n_hit + 1.0


def z2(realized, var, es, alpha: float) -> float:
    r, v, e = (np.asarray(x, float) for x in (realized, var, es))
    s, _ = _tail_ratio_sum(r, v, e)
    return s / (r.size * alpha) + 1.0


def pvalue_by_simulation(
    stat_obs: float,
    sims: np.ndarray,
    var: np.ndarray,
    es: np.ndarray,
    alpha: float,
    *,
    which: str = "z2",
) -> float:
    """One-sided p-value: P(Z <= stat_obs) under H0, from ``sims`` — a
    ``(n_boot, n_days)`` array of returns drawn from the model's predictive
    distribution for each day. Small p -> reject (ES too optimistic)."""
    var = np.asarray(var, float)
    es = np.asarray(es, float)
    fn = z1 if which == "z1" else (lambda r, v, e: z2(r, v, e, alpha))
    null = np.array([fn(sims[b], var, es) for b in range(sims.shape[0])])
    null = null[np.isfinite(null)]
    if null.size == 0:
        return np.nan
    return float((null <= stat_obs + 1e-12).mean())
