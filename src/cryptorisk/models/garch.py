"""Single-regime GARCH family via ``arch`` (V2_PLAN §3).

* ``GarchT``   - GARCH(1,1)-t
* ``GjrGarchT`` - GJR-GARCH(1,1)-t  (leverage: ``o=1``)
* ``EgarchT``  - EGARCH(1,1)-t

Returns must be scaled x100 for the optimiser; scaled back afterwards. The
predictive distribution is location-scale with a **standardized** Student-t
(see ``_dist.student_t_z``). On a failed fit (or ``nu <= 2``) the model falls
back to the empirical quantile of the window; ``FALLBACK_COUNT`` tracks how
often, and the study reports it.
"""

from __future__ import annotations

import numpy as np

from cryptorisk.models._dist import student_t_z
from cryptorisk.models.base import Context, EmpiricalDist, ParametricDist, PredictiveDist

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


def _fallback(r: np.ndarray) -> PredictiveDist:
    global FALLBACK_COUNT
    FALLBACK_COUNT += 1
    return EmpiricalDist(r)


class _ArchModel:
    """Shared driver. Subclasses set ``vol``, ``o`` and ``name``."""

    vol = "GARCH"
    o = 0
    name = "GARCH-t"

    def fit_predict(self, ctx: Context) -> PredictiveDist:
        r = ctx.returns
        if not _ARCH or r.size < 100:
            return _fallback(r)
        try:
            am = arch_model(
                r * _SCALE, mean="Constant", vol=self.vol, p=1, o=self.o, q=1,
                dist="t", rescale=False,
            )
            res = am.fit(disp="off", show_warning=False)
            fc = res.forecast(horizon=1, reindex=False)
            sigma_next = float(np.sqrt(fc.variance.iloc[-1, 0])) / _SCALE
            mu_next = float(res.params.get("mu", 0.0)) / _SCALE
            nu = float(res.params.get("nu", 8.0))
            if not np.isfinite(sigma_next) or sigma_next <= 0 or nu <= 2.05:
                return _fallback(r)
            ppf, cdf, es = student_t_z(nu)
            return ParametricDist(loc=mu_next, scale=sigma_next, z_ppf=ppf, z_cdf=cdf, z_es=es)
        except Exception:  # noqa: BLE001 - any arch/optimiser failure -> empirical
            return _fallback(r)


class GarchT(_ArchModel):
    vol, o, name = "GARCH", 0, "GARCH-t"


class GjrGarchT(_ArchModel):
    vol, o, name = "GARCH", 1, "GJR-GARCH-t"


class EgarchT(_ArchModel):
    vol, o, name = "EGARCH", 1, "EGARCH-t"
