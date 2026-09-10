"""GARCH-EVT (McNeil & Frey, 2000).

Two steps: a GARCH(1,1)-t filter removes the volatility dynamics; a GPD is fitted
by Peaks-Over-Threshold to the lower tail of the standardized residuals. The
one-step forecast is ``mu_{t+1} + sigma_{t+1} * z``, with ``z`` drawn from the
GPD tail (see :class:`cryptorisk.models._gpd.GpdTailDist`).

Ported from v1's corrected ``evt.py``. Falls back to a plain empirical
distribution of the window when ``arch`` is unavailable or the filter fails.
"""

from __future__ import annotations

import numpy as np

from cryptorisk.models._gpd import GpdTailDist
from cryptorisk.models.base import Context, EmpiricalDist, PredictiveDist

try:
    from arch import arch_model

    _ARCH = True
except ImportError:  # pragma: no cover
    _ARCH = False

FALLBACK_COUNT = 0
_SCALE = 100.0


def reset_fallback_count() -> None:
    global FALLBACK_COUNT
    FALLBACK_COUNT = 0


class GarchEVT:
    name = "GARCH-EVT"

    def __init__(self, threshold_q: float = 0.90):
        self.threshold_q = threshold_q

    def fit_predict(self, ctx: Context) -> PredictiveDist:
        global FALLBACK_COUNT
        r = ctx.returns
        if not _ARCH or r.size < 150:
            FALLBACK_COUNT += 1
            return EmpiricalDist(r)
        try:
            res = arch_model(
                r * _SCALE, mean="Constant", vol="GARCH", p=1, q=1, dist="t", rescale=False
            ).fit(disp="off", show_warning=False)
            z = np.asarray(res.std_resid, float)
            z = z[np.isfinite(z)]
            fc = res.forecast(horizon=1, reindex=False)
            sigma_next = float(np.sqrt(fc.variance.iloc[-1, 0])) / _SCALE
            mu_next = float(res.params.get("mu", 0.0)) / _SCALE
            if z.size < 150 or not np.isfinite(sigma_next) or sigma_next <= 0:
                FALLBACK_COUNT += 1
                return EmpiricalDist(r)
            return GpdTailDist(z, loc=mu_next, scale=sigma_next, threshold_q=self.threshold_q)
        except Exception:  # noqa: BLE001
            FALLBACK_COUNT += 1
            return EmpiricalDist(r)
