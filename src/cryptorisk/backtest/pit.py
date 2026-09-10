"""Probability Integral Transform diagnostics (V2_PLAN §5.6).

``u_t = F_t(r_t)``; under a correctly specified model ``u_t ~ iid U(0, 1)``.

* ``pit_values`` — clean the PIT series (finite, clipped away from {0, 1}).
* ``berkowitz`` — Berkowitz (2001) LR test. Transform ``z_t = Phi^{-1}(u_t)``;
  under H0 ``z ~ iid N(0, 1)``. Fit an AR(1) ``z_t = mu + rho z_{t-1} + eps``
  and LR-test H0: ``mu = 0, rho = 0, sigma^2 = 1`` (~ chi2(3)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

_EPS = 1e-6


def pit_values(pit) -> np.ndarray:
    u = np.asarray(pit, float)
    u = u[np.isfinite(u)]
    return np.clip(u, _EPS, 1.0 - _EPS)


@dataclass(frozen=True)
class BerkowitzResult:
    statistic: float
    p_value: float
    mu: float
    rho: float
    sigma2: float


def berkowitz(pit) -> BerkowitzResult:
    u = pit_values(pit)
    if u.size < 20:
        return BerkowitzResult(np.nan, np.nan, np.nan, np.nan, np.nan)
    z = stats.norm.ppf(u)

    y, x = z[1:], z[:-1]
    n = y.size
    xc, yc = x - x.mean(), y - y.mean()
    rho = float(xc @ yc / (xc @ xc)) if xc @ xc > 0 else 0.0
    mu = float(y.mean() - rho * x.mean())
    resid = y - mu - rho * x
    s2 = float(resid @ resid / n)
    s2 = max(s2, 1e-12)

    ll1 = -0.5 * (n * np.log(2 * np.pi) + n * np.log(s2) + (resid @ resid) / s2)
    ll0 = -0.5 * (n * np.log(2 * np.pi) + (y @ y))
    lr = float(-2.0 * (ll0 - ll1))
    return BerkowitzResult(lr, float(stats.chi2.sf(lr, 3)), mu, rho, s2)
