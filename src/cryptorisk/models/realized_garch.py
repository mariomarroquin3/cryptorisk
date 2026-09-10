"""Realized GARCH(1,1) - log-linear (Hansen, Huang & Shek, 2012), V2_PLAN §3.

    r_t   = sqrt(h_t) * z_t ,           z_t ~ N(0, 1)
    log h_t = omega + beta log h_{t-1} + gamma log x_{t-1}
    log x_t = xi + phi log h_t + tau1 z_t + tau2 (z_t^2 - 1) + u_t ,  u_t ~ N(0, s_u^2)

``x`` is the realized variance (``ctx.realized["rv"]``). Joint Gaussian MLE over
(omega, beta, gamma, xi, phi, tau1, tau2, s_u). The measurement equation plus the
leverage function tau(z) let the model use the realized measure while staying
identified. Forecast: h_{t+1} = exp(omega + beta log h_t + gamma log x_t).

Falls back to the empirical quantile when no realized block is present or the
optimiser fails.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scipy.signal import lfilter

from cryptorisk.models._dist import student_t_z
from cryptorisk.models._util import ffill
from cryptorisk.models.base import Context, EmpiricalDist, ParametricDist, PredictiveDist


def _filter(theta, r, logx):
    """log h_t = omega + beta log h_{t-1} + gamma log x_{t-1} -- an AR(1) in
    log h with a time-varying input, solved exactly with ``lfilter``."""
    omega, beta, gamma, xi, phi, tau1, tau2, log_su2 = theta
    su2 = np.exp(log_su2)
    inp = np.empty(r.size)
    inp[0] = np.log(np.var(r) + 1e-12)          # log h_0
    inp[1:] = omega + gamma * logx[:-1]
    logh = lfilter([1.0], [1.0, -beta], inp, zi=[0.0])[0]
    logh = np.clip(logh, -50, 50)
    z = r / np.sqrt(np.exp(logh))
    u = logx - (xi + phi * logh + tau1 * z + tau2 * (z**2 - 1))
    return logh, z, u, su2


class RealizedGARCH:
    name = "Realized-GARCH"

    def fit_predict(self, ctx: Context) -> PredictiveDist:
        r = np.asarray(ctx.returns, float)
        if not ctx.realized or "rv" not in ctx.realized:
            return EmpiricalDist(r)
        rv = np.asarray(ctx.realized["rv"], float)
        rv = np.where(np.isfinite(rv) & (rv > 0), rv, np.nan)
        if np.isnan(rv).mean() > 0.2 or r.size < 120:
            return EmpiricalDist(r)
        logx = np.log(ffill(rv))[: r.size]

        x0 = np.array([0.0, 0.6, 0.4, 0.0, 1.0, -0.05, 0.05, np.log(0.3)])
        bounds = [(-5, 5), (0.0, 0.999), (0.0, 0.999), (-10, 10), (0.1, 3.0),
                  (-1, 1), (-1, 1), (np.log(1e-4), np.log(5.0))]

        def nll(theta):
            if theta[1] + theta[2] >= 1.02:
                return 1e10
            logh, z, u, su2 = _filter(theta, r, logx)
            ll_r = -0.5 * np.sum(logh + z**2)
            ll_x = -0.5 * np.sum(np.log(2 * np.pi * su2) + u**2 / su2)
            v = -(ll_r + ll_x)
            return v if np.isfinite(v) else 1e10

        try:
            res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 400})
            th = res.x if res.success else x0
        except Exception:  # noqa: BLE001
            return EmpiricalDist(r)

        logh, z, _, _ = _filter(th, r, logx)
        logh_next = th[0] + th[1] * logh[-1] + th[2] * logx[-1]
        sigma_next = float(np.sqrt(np.exp(np.clip(logh_next, -50, 50))))
        if not np.isfinite(sigma_next) or sigma_next <= 0 or sigma_next > 20 * r.std():
            return EmpiricalDist(r)

        zf = z[np.isfinite(z)]
        nu = float(np.clip(stats.t.fit(zf, floc=0)[0], 3.0, 50.0)) if zf.size > 50 else 8.0
        ppf, cdf, es = student_t_z(nu)
        return ParametricDist(loc=0.0, scale=sigma_next, z_ppf=ppf, z_cdf=cdf, z_es=es)

