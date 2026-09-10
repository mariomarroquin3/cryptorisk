"""Per-asset volatility filter for the copula portfolio model.

GARCH(1,1)-t via ``arch`` (the single-asset FHS/GARCH pattern), returning the
one-step (mu, sigma), the standardized residuals for the copula step, and the
fitted degrees of freedom. Falls back to EWMA + empirical residuals when
``arch`` is unavailable or the fit degenerates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from arch import arch_model

    _ARCH = True
except ImportError:  # pragma: no cover
    _ARCH = False

_SCALE = 100.0


@dataclass(frozen=True)
class MarginalFit:
    mu_next: float          # one-step conditional mean of the log-return
    sigma_next: float       # one-step conditional s.d.
    z_resid: np.ndarray     # standardized residuals over the window (finite)
    nu: float               # fitted Student-t d.o.f. (np.inf for the EWMA fallback)
    ok: bool                # False if the GARCH fit was rejected


def _ewma_fallback(r: np.ndarray, lam: float = 0.94) -> MarginalFit:
    """Used only when the GARCH fit is unavailable/degenerate. ``sig[-1]`` is
    the next-day EWMA vol; ``z`` is standardised contemporaneously (``sig[i]``
    has absorbed ``r[i]``), which is fine here -- the residuals only feed the
    copula's rank/dependence estimate, not a forecast."""
    s2 = float(np.mean(r[: min(30, r.size)] ** 2))
    sig = np.empty(r.size)
    for i, x in enumerate(r):
        s2 = lam * s2 + (1.0 - lam) * float(x) ** 2
        sig[i] = np.sqrt(max(s2, 1e-18))
    mu = float(np.mean(r))
    z = (r - mu) / sig
    return MarginalFit(mu, float(sig[-1]), z[np.isfinite(z)], np.inf, ok=False)


def fit_marginal(returns: np.ndarray) -> MarginalFit:
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    if not _ARCH or r.size < 100:
        return _ewma_fallback(r)
    try:
        res = arch_model(
            r * _SCALE, mean="Constant", vol="GARCH", p=1, q=1, dist="t", rescale=False
        ).fit(disp="off", show_warning=False)
        z = np.asarray(res.std_resid, float)
        z = z[np.isfinite(z)]
        fc = res.forecast(horizon=1, reindex=False)
        sigma_next = float(np.sqrt(fc.variance.iloc[-1, 0])) / _SCALE
        mu_next = float(res.params.get("mu", 0.0)) / _SCALE
        nu = float(res.params.get("nu", 8.0))
        sd = float(r.std())
        if (
            z.size < 100
            or not np.isfinite(sigma_next)
            or not (0.05 * sd < sigma_next < 20.0 * sd)
            or nu <= 2.05
        ):
            return _ewma_fallback(r)
        return MarginalFit(mu_next, sigma_next, z, nu, ok=True)
    except Exception:  # noqa: BLE001
        return _ewma_fallback(r)
