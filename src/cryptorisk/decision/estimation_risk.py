"""Estimation-risk bands on VaR / ES (V2_PLAN §6, closing a §9 limitation).

The headline backtests report VaR/ES at the parameter *point* estimate. The
estimator has a sampling distribution, so the forecast does too; ignoring that
makes a reported interval too tight and a capital number too low
(Christoffersen & Goncalves, 2005).

This module quantifies the effect on the **final estimation window** (the window
that would forecast the next day) for three archetypes:

* ``garch_t_band``  -- parameter uncertainty only: draw
  ``theta* ~ N(theta_hat, Sigma_hat)`` from the ``arch`` GARCH(1,1)-t fit and
  recompute the one-step sigma with ``forecast(params=theta*)`` (no refit),
  then the standardized-t VaR/ES.
* ``hs_band``       -- stationary block bootstrap of the window; empirical
  quantile / tail mean per resample (Historical Simulation).
* ``fhs_band``      -- GARCH(1,1)-normal: parameter draw for the vol path plus a
  residual resample for the tail (Filtered HS).

Each returns an :class:`EstimationRiskBand`: point, bootstrap mean / s.e., the
5th/95th-percentile interval, and a **prudent** ES (the 5th percentile of the
ES draws -- the conservative tail). :func:`capital_addon` turns the gap between
the point and prudent ES into currency, to sit next to the model-risk add-on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cryptorisk.decision.capital import es_capital
from cryptorisk.models._dist import student_t_z

try:
    from arch import arch_model

    _ARCH = True
except ImportError:  # pragma: no cover
    _ARCH = False

_SCALE = 100.0


@dataclass(frozen=True)
class EstimationRiskBand:
    estimator: str
    alpha: float
    n_draws: int
    var_point: float
    es_point: float
    var_mean: float
    es_mean: float
    var_se: float
    es_se: float
    var_lo: float          # 5th percentile of the VaR draws (most negative side is lo)
    var_hi: float           # 95th percentile
    es_lo: float
    es_hi: float
    es_prudent: float       # 5th percentile of the ES draws -- the conservative tail

    def es_widening(self) -> float:
        """Prudent |ES| minus point |ES|, as a positive return fraction."""
        return abs(self.es_prudent) - abs(self.es_point)


def _summarise(estimator: str, alpha: float, var_point: float, es_point: float,
               var_draws: np.ndarray, es_draws: np.ndarray) -> EstimationRiskBand:
    var_draws = var_draws[np.isfinite(var_draws)]
    es_draws = es_draws[np.isfinite(es_draws)]
    if var_draws.size < 20 or es_draws.size < 20:
        nan = float("nan")
        return EstimationRiskBand(estimator, alpha, int(var_draws.size), var_point,
                                  es_point, nan, nan, nan, nan, nan, nan, nan, nan, nan)
    return EstimationRiskBand(
        estimator=estimator,
        alpha=alpha,
        n_draws=int(min(var_draws.size, es_draws.size)),
        var_point=float(var_point),
        es_point=float(es_point),
        var_mean=float(var_draws.mean()),
        es_mean=float(es_draws.mean()),
        var_se=float(var_draws.std(ddof=1)),
        es_se=float(es_draws.std(ddof=1)),
        var_lo=float(np.quantile(var_draws, 0.05)),
        var_hi=float(np.quantile(var_draws, 0.95)),
        es_lo=float(np.quantile(es_draws, 0.05)),
        es_hi=float(np.quantile(es_draws, 0.95)),
        es_prudent=float(np.quantile(es_draws, 0.05)),
    )


# --------------------------------------------------------------------------- #
def _empirical_var_es(sample: np.ndarray, alpha: float) -> tuple[float, float]:
    q = float(np.quantile(sample, alpha))
    tail = sample[sample <= q]
    return q, (float(tail.mean()) if tail.size else q)


def hs_band(
    returns, alphas: list[float], *, n_boot: int = 2000, block_len: int = 10,
    seed: int | None = None,
) -> list[EstimationRiskBand]:
    """Stationary block bootstrap of the estimation window."""
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    n = r.size
    rng = np.random.default_rng(seed)
    p = 1.0 / max(block_len, 1.0)

    idx = np.empty((n_boot, n), dtype=np.int64)
    for b in range(n_boot):
        cur = rng.integers(n)
        for t in range(n):
            if t and rng.random() < p:
                cur = rng.integers(n)
            idx[b, t] = cur
            cur = (cur + 1) % n
    resamples = r[idx]  # (n_boot, n)

    out = []
    for a in alphas:
        vp, ep = _empirical_var_es(r, a)
        qs = np.quantile(resamples, a, axis=1)
        vd = np.empty(n_boot)
        ed = np.empty(n_boot)
        for b in range(n_boot):
            row = resamples[b]
            tail = row[row <= qs[b]]
            vd[b] = qs[b]
            ed[b] = tail.mean() if tail.size else qs[b]
        out.append(_summarise("HS", a, vp, ep, vd, ed))
    return out


def _fit_garch(r: np.ndarray, dist: str):
    am = arch_model(r * _SCALE, mean="Constant", vol="GARCH", p=1, q=1, dist=dist, rescale=False)
    res = am.fit(disp="off", show_warning=False)
    return am, res


def _valid_garch_params(theta: np.ndarray, *, has_nu: bool) -> bool:
    omega, a1, b1 = theta[1], theta[2], theta[3]
    if not (np.isfinite(theta).all() and omega > 0 and a1 >= 0 and b1 >= 0 and a1 + b1 < 0.999):
        return False
    return not (has_nu and theta[-1] <= 2.05)


def garch_t_band(
    returns, alphas: list[float], *, n_draws: int = 2500, seed: int | None = None,
) -> list[EstimationRiskBand]:
    """Parameter-uncertainty band for GARCH(1,1)-t: draw the params from the
    fitted asymptotic covariance and re-forecast (no refit)."""
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    if not _ARCH or r.size < 100:
        return [EstimationRiskBand("GARCH-t", a, 0, *([float("nan")] * 11)) for a in alphas]
    am, res = _fit_garch(r, "t")
    th = res.params.to_numpy(float)
    cov = np.asarray(res.param_cov, float)
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(th, cov, size=n_draws)

    def _one(theta: np.ndarray) -> tuple[float, float, float] | None:
        if not _valid_garch_params(theta, has_nu=True):
            return None
        try:
            fc = am.forecast(params=theta, horizon=1, reindex=False)
            sig = float(np.sqrt(fc.variance.iloc[-1, 0])) / _SCALE
        except Exception:  # noqa: BLE001
            return None
        mu = float(theta[0]) / _SCALE
        nu = float(theta[-1])
        if not np.isfinite(sig) or sig <= 0:
            return None
        return mu, sig, nu

    pt = _one(th)
    if pt is None:
        return [EstimationRiskBand("GARCH-t", a, 0, *([float("nan")] * 11)) for a in alphas]
    mu0, sig0, nu0 = pt
    ppf0, _cdf0, es0 = student_t_z(nu0)

    rows: list[tuple[float, float, float]] = []
    for d in draws:
        got = _one(d)
        if got is None:
            continue
        rows.append(got)
    mus = np.array([m for m, _, _ in rows])
    sigs = np.array([s for _, s, _ in rows])
    nus = np.array([n for _, _, n in rows])

    out = []
    for a in alphas:
        vp = mu0 + sig0 * float(ppf0(a))
        ep = mu0 + sig0 * float(es0(a))
        vd = np.empty(len(rows))
        ed = np.empty(len(rows))
        for i, (m, s, nu) in enumerate(zip(mus, sigs, nus, strict=True)):
            pf, _cf, ef = student_t_z(nu)
            vd[i] = m + s * float(pf(a))
            ed[i] = m + s * float(ef(a))
        out.append(_summarise("GARCH-t", a, vp, ep, vd, ed))
    return out


def fhs_band(
    returns, alphas: list[float], *, n_draws: int = 2500, seed: int | None = None,
) -> list[EstimationRiskBand]:
    """FHS: GARCH(1,1)-normal parameter draw for the vol path, plus a residual
    resample for the tail."""
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    if not _ARCH or r.size < 100:
        return [EstimationRiskBand("FHS", a, 0, *([float("nan")] * 11)) for a in alphas]
    am, res = _fit_garch(r, "normal")
    th = res.params.to_numpy(float)
    cov = np.asarray(res.param_cov, float)
    z = np.asarray(res.std_resid, float)
    z = z[np.isfinite(z)]
    if z.size < 100:
        return [EstimationRiskBand("FHS", a, 0, *([float("nan")] * 11)) for a in alphas]

    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(th, cov, size=n_draws)

    def _sigma_mu(theta: np.ndarray) -> tuple[float, float] | None:
        if not _valid_garch_params(theta, has_nu=False):
            return None
        try:
            fc = am.forecast(params=theta, horizon=1, reindex=False)
            sig = float(np.sqrt(fc.variance.iloc[-1, 0])) / _SCALE
        except Exception:  # noqa: BLE001
            return None
        if not np.isfinite(sig) or sig <= 0:
            return None
        return float(theta[0]) / _SCALE, sig

    pt = _sigma_mu(th)
    if pt is None:
        return [EstimationRiskBand("FHS", a, 0, *([float("nan")] * 11)) for a in alphas]
    mu0, sig0 = pt
    q0 = {a: float(np.quantile(z, a)) for a in alphas}
    tail0 = {a: (float(z[z <= q0[a]].mean()) if (z <= q0[a]).any() else q0[a]) for a in alphas}

    sm = [got for d in draws if (got := _sigma_mu(d)) is not None]
    out = []
    for a in alphas:
        vp = mu0 + sig0 * q0[a]
        ep = mu0 + sig0 * tail0[a]
        vd = np.empty(len(sm))
        ed = np.empty(len(sm))
        for i, (mu, sig) in enumerate(sm):
            zb = rng.choice(z, size=z.size, replace=True)
            qb = float(np.quantile(zb, a))
            tb = zb[zb <= qb]
            vd[i] = mu + sig * qb
            ed[i] = mu + sig * (tb.mean() if tb.size else qb)
        out.append(_summarise("FHS", a, vp, ep, vd, ed))
    return out


# --------------------------------------------------------------------------- #
def capital_addon(
    band: EstimationRiskBand, *, liquidity_horizon: int, multiplier: float, notional: float,
) -> float:
    """Extra capital from moving the 97.5% ES from its point estimate to the
    prudent (5th-percentile) draw: ``es_capital(prudent) - es_capital(point)``.
    Non-negative by construction (prudent is the more negative tail)."""
    if not np.isfinite(band.es_point) or not np.isfinite(band.es_prudent):
        return float("nan")
    c_point = es_capital(
        band.es_point, liquidity_horizon=liquidity_horizon, multiplier=multiplier, notional=notional
    )
    c_prudent = es_capital(
        band.es_prudent, liquidity_horizon=liquidity_horizon, multiplier=multiplier, notional=notional
    )
    return float(max(c_prudent - c_point, 0.0))
