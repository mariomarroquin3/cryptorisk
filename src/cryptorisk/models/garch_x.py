"""GARCH-X: GARCH(1,1)-t with an exogenous term in the variance equation
(V2_PLAN §3; Engle 2002; Han & Kristensen 2014).

    h_t = omega + alpha * r_{t-1}^2 + beta * h_{t-1} + gamma * x_{t-1}

with ``x`` the lagged realized variance (``ctx.realized["rv"]``) and standardized
Student-t innovations. Hand-coded MLE (``arch`` has no variance-equation exog in
its simple API). Falls back to a plain GARCH(1,1)-t (gamma = 0) when no realized
block is available, and to the empirical quantile if the optimiser fails.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import gammaln

from cryptorisk.models._dist import student_t_z
from cryptorisk.models._util import ffill
from cryptorisk.models.base import Context, EmpiricalDist, ParametricDist, PredictiveDist

_S = 100.0  # scale returns; x (a variance) scales by _S**2


def _std_t_logpdf(z: np.ndarray, nu: float) -> np.ndarray:
    s2 = (nu - 2.0) / nu
    c = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * nu * s2)
    return c - (nu + 1) / 2 * np.log1p(z**2 / (s2 * nu))


def _recursion(theta, r2, x):
    """h_t = omega + alpha r2_{t-1} + beta h_{t-1} + gamma x_{t-1} -- AR(1) in h
    with a time-varying input, solved with ``lfilter``."""
    omega, alpha, beta, gamma, _nu = theta
    inp = np.empty(r2.size)
    inp[0] = r2.mean()                          # h_0
    news = alpha * r2[:-1] + (gamma * x[:-1] if x is not None else 0.0)
    inp[1:] = omega + news
    h = lfilter([1.0], [1.0, -beta], inp, zi=[0.0])[0]
    return np.clip(np.nan_to_num(h, nan=1e-12), 1e-12, None)


class GarchX:
    name = "GARCH-X"

    def fit_predict(self, ctx: Context) -> PredictiveDist:
        r = ctx.returns * _S
        r2 = r**2
        x = None
        if ctx.realized and "rv" in ctx.realized:
            xv = np.asarray(ctx.realized["rv"], float) * _S**2
            if np.isfinite(xv).mean() > 0.8:
                x = ffill(np.where(np.isfinite(xv) & (xv > 0), xv, np.nan))[: r.size]

        v0 = float(r2.mean())
        x0 = np.array([0.05 * v0, 0.08, 0.88, 0.02 if x is not None else 0.0, 7.0])
        bounds = [(1e-8, 10 * v0), (0.0, 0.5), (0.0, 0.999),
                  (0.0, 5.0) if x is not None else (0.0, 0.0), (2.1, 50.0)]

        def nll(theta):
            if theta[1] + theta[2] >= 1.0:
                return 1e10
            h = _recursion(theta, r2, x)
            z = r / np.sqrt(h)
            return -float(np.sum(_std_t_logpdf(z, theta[4]) - 0.5 * np.log(h)))

        try:
            res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 300})
            th = res.x if res.success else x0
        except Exception:  # noqa: BLE001
            return EmpiricalDist(ctx.returns)

        h = _recursion(th, r2, x)
        h_next = th[0] + th[1] * r2[-1] + th[2] * h[-1] + th[3] * (x[-1] if x is not None else 0.0)
        sigma_next = float(np.sqrt(max(h_next, 1e-12))) / _S
        nu = float(np.clip(th[4], 3.0, 50.0))
        if not np.isfinite(sigma_next) or sigma_next <= 0 or sigma_next > 20 * ctx.returns.std():
            return EmpiricalDist(ctx.returns)
        ppf, cdf, es = student_t_z(nu)
        return ParametricDist(loc=0.0, scale=sigma_next, z_ppf=ppf, z_cdf=cdf, z_es=es)

