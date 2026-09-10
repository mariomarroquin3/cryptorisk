"""Filtered Historical Simulation (Barone-Adesi, Giannopoulos & Vosper, 1999).

A GARCH(1,1) filter removes the volatility dynamics; the one-step VaR/ES are the
empirical quantile / tail mean of the standardized residuals, rescaled by
sigma_{t+1}. The residual distribution is left non-parametric - that is the
whole point.
"""

from __future__ import annotations

import numpy as np

from cryptorisk.models.base import Context, EmpiricalDist, PredictiveDist

try:
    from arch import arch_model

    _ARCH = True
except ImportError:  # pragma: no cover
    _ARCH = False

_SCALE = 100.0


class FHS:
    name = "FHS"

    def fit_predict(self, ctx: Context) -> PredictiveDist:
        r = ctx.returns
        if not _ARCH or r.size < 100:
            return EmpiricalDist(r)
        try:
            res = arch_model(
                r * _SCALE, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False
            ).fit(disp="off", show_warning=False)
            z = np.asarray(res.std_resid, float)
            z = z[np.isfinite(z)]
            fc = res.forecast(horizon=1, reindex=False)
            sigma_next = float(np.sqrt(fc.variance.iloc[-1, 0])) / _SCALE
            mu_next = float(res.params.get("mu", 0.0)) / _SCALE
            if z.size < 100 or not np.isfinite(sigma_next) or sigma_next <= 0:
                return EmpiricalDist(r)
            return EmpiricalDist(sample=z, scale=sigma_next, loc=mu_next)
        except Exception:  # noqa: BLE001
            return EmpiricalDist(r)
