"""Realized Stochastic Volatility (Heston-style CIR variance, UKF quasi-MLE,
GJR-style leverage).

    dV_t = kappa*(theta - V_t) dt + gamma*I(z_{t-1}<0)*(r_{t-1}-mu)^2
           + xi*sqrt(V_t) dW_t                          (CIR, dt=1, Euler
                                                          full truncation)
    r_t  = mu + sqrt(V_t) * z_t,  z_t ~ standardized Student-t(nu)
    RV_t = exp(zeta) * V_t^phi * exp(u_t),  u_t ~ N(0, sigma_u^2)

Unlike Realized-GARCH's log-linear AR(1) state, ``V_t`` follows the literal
square-root diffusion behind Heston's stochastic-volatility SDE -- that
nonlinear, state-dependent-noise transition is exactly why this needs an
Unscented Kalman Filter rather than a closed-form linear filter (a plain KF
assumes linear-Gaussian dynamics throughout). Estimated by quasi-MLE, standard
for approximate-filter SV estimation (e.g. Javaheri, Lautier & Galli, 2003),
but the two observation channels are scored two different ways:

* **Return channel -- exact density, no state update.** An earlier version
  routed the return through the Harvey-Ruiz-Shephard (1994) trick
  (``log((r_t-mu)^2) approx log V_t + E[log z^2]``), which only matches the
  first two moments of ``log z^2`` -- itself never Gaussian even after
  matching mean/variance to the fitted ``nu`` (it's a log-F(1,nu) variate,
  with its own skew a 2-moment Gaussian update can't see). That approximation
  turned out to be the dominant remaining source of miscalibration after
  fixing its moments for fat tails (see git history), so the return no
  longer updates the *filtered state* at all. It still fully determines
  ``mu``, ``nu`` and (via ``theta``/``kappa``/``xi``) the shape of the ``V``
  path, through its own **exact** quasi-likelihood contribution added
  directly to the total log-likelihood: ``log f_Z((r_t-mu)/sqrt(V_pred); nu)
  - 0.5*log(V_pred)``, the honest log-density of ``r_t | V_pred`` under a
  standardized Student-t -- no moment-matching, no approximation of the
  observation model itself (``_student_t_ll``).
* **Realized-measure channel -- the one UKF update.** ``y_t = log RV_t = zeta
  + phi*log V_t + u_t``, mirroring Realized-GARCH's measurement equation
  (Hansen, Huang & Shek, 2012) with a fitted (not fixed-at-1) slope ``phi``.
  RV is a low-noise, already-close-to-Gaussian-in-log-space measurement of
  V_t (that's what makes Realized-GARCH itself work), so this is the only
  place an approximate Gaussian UKF update is actually a good fit -- the
  filtered ``V`` path is driven entirely by RV plus the CIR's own dynamics.

The (V, eta)-augmented sigma points in ``_predict`` propagate the CIR's
``sqrt(V)*eta`` noise term through the unscented transform to 2nd order
without linearizing it.

**Leverage.** A *same-day* correlation between the return shock and the
volatility shock (Heston's ``rho``) would need the sign of ``z_t`` before
the day's own predict step, which does not exist yet. Instead this uses a
GJR-GARCH-style *lagged* asymmetric term in the CIR drift: a down day at
``t-1`` pushes ``V_t`` up by ``gamma * (r_{t-1}-mu)^2``, on top of the usual
mean reversion. This is causally clean (no lookahead: the predict step for
day ``t`` only uses information through ``t-1``) and reuses a well-tested
asymmetric-response shape already in the suite
(``models/garch.py::GjrGarchT``), rather than attempting a same-day
correlated-noise UKF.

Falls back to the empirical quantile under the same conditions as
Realized-GARCH (no/too little realized data, or a degenerate fit).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit
from scipy.optimize import minimize

from cryptorisk.models._gpd import GpdTailDist
from cryptorisk.models._util import ffill
from cryptorisk.models.base import Context, EmpiricalDist, PredictiveDist

_EPS = 1e-10
# Belt-and-suspenders ceiling on the filter's own state-variance estimate p
# (units of V^2). kappa < 2 (enforced via fit_predict's bounds) keeps the
# linearized recursion from diverging on its own, but this still guards
# against overflow/NaN if an optimizer step briefly probes a bad corner.
_P_CEIL = 1e6

# Scaled-UKF tuning (Julier & Uhlmann, 1997; Wan & van der Merwe, 2000).
# alpha=1 keeps the sigma-point spread wide enough for the sqrt(V)
# nonlinearity; beta=2 is optimal for a Gaussian prior.
_UKF_ALPHA, _UKF_BETA, _UKF_KAPPA = 1.0, 2.0, 0.0


def _ukf_weights(n: int) -> tuple[float, np.ndarray, np.ndarray]:
    """lambda, mean weights, covariance weights for an n-dim sigma-point set."""
    lam = _UKF_ALPHA**2 * (n + _UKF_KAPPA) - n
    wm = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
    wc = wm.copy()
    wm[0] = lam / (n + lam)
    wc[0] = wm[0] + (1 - _UKF_ALPHA**2 + _UKF_BETA)
    return lam, wm, wc


_LAM1, _WM1, _WC1 = _ukf_weights(1)  # additive-noise measurement update
_LAM2, _WM2, _WC2 = _ukf_weights(2)  # (V, eta)-augmented state transition

# The scaled UKF's four off-center sigma points always share one weight (only
# the center point differs), so the whole sum collapses to scalar arithmetic
# on named points -- no numpy array allocation needed in the per-day loop
# below. Tiny (3- or 5-element) numpy arrays allocated ~900-2600 times per
# optimizer call were the actual cost driver (measured ~60ms/window pass
# before this rewrite; numpy's fixed per-call overhead dominates at this
# array size), not the UKF math itself.
_WM2_0, _WM2_OTHER = float(_WM2[0]), float(_WM2[1])
_WC2_0, _WC2_OTHER = float(_WC2[0]), float(_WC2[1])
_WM1_0, _WM1_OTHER = float(_WM1[0]), float(_WM1[1])
_WC1_0, _WC1_OTHER = float(_WC1[0]), float(_WC1[1])
_SPREAD2_FACTOR = 2 + _LAM2
_SPREAD1_FACTOR = 1 + _LAM1


@njit(cache=True)
def _predict(v: float, p: float, kappa: float, theta: float, xi: float, lev: float) -> tuple[float, float]:
    """CIR transition (Euler, full truncation) via a (V, eta)-augmented UKF
    sigma-point set, in plain scalar arithmetic (see module note above).
    ``lev`` is the GJR-style lagged leverage kick
    (``gamma * I(z_{t-1}<0) * (r_{t-1}-mu)^2``), already resolved by the
    caller -- a plain additive drift term, same for every sigma point."""
    spread_v = math.sqrt(max(_SPREAD2_FACTOR * p, 0.0))
    spread_e = math.sqrt(_SPREAD2_FACTOR)

    # f(vv, ee) = vv + kappa*(theta - max(vv,0)) + lev + xi*sqrt(max(vv,0))*ee,
    # inlined (no nested-function object per call -- this runs ~900-2600
    # times per optimizer NLL evaluation).
    v_plus = v if v > 0.0 else 0.0
    sqrt_v_plus = math.sqrt(v_plus)
    vp1 = v + spread_v
    vp1_plus = vp1 if vp1 > 0.0 else 0.0
    vp3 = v - spread_v
    vp3_plus = vp3 if vp3 > 0.0 else 0.0

    f0 = v + kappa * (theta - v_plus) + lev
    f1 = vp1 + kappa * (theta - vp1_plus) + lev
    f2 = v + kappa * (theta - v_plus) + lev + xi * sqrt_v_plus * spread_e
    f3 = vp3 + kappa * (theta - vp3_plus) + lev
    f4 = v + kappa * (theta - v_plus) + lev - xi * sqrt_v_plus * spread_e
    v_pred = _WM2_0 * f0 + _WM2_OTHER * (f1 + f2 + f3 + f4)
    p_pred = _WC2_0 * (f0 - v_pred) ** 2 + _WC2_OTHER * (
        (f1 - v_pred) ** 2 + (f2 - v_pred) ** 2 + (f3 - v_pred) ** 2 + (f4 - v_pred) ** 2
    )
    # Full truncation: the sigma-point transform's own weighted mean can land
    # below 0 even though every point it averages was pushed through vp=max(.,0)
    # -- the CIR state itself, not just intermediate sqrt/log evaluations,
    # must be clamped, or a negative v propagates forward almost unchanged
    # (vp=0 barely moves) and the filter gets stuck there permanently.
    return max(v_pred, _EPS), min(max(p_pred, _EPS), _P_CEIL)


@njit(cache=True)
def _update(
    v: float, p: float, y_obs: float, r_noise: float, bias: float, phi: float
) -> tuple[float, float, float]:
    """Scalar UKF measurement update through ``h(V) = bias + phi*log(max(V, eps))``.
    Returns ``(v_post, p_post, log-likelihood contribution)``."""
    spread = math.sqrt(max(_SPREAD1_FACTOR * p, 0.0))
    y0 = bias + phi * math.log(max(v, _EPS))
    y1 = bias + phi * math.log(max(v + spread, _EPS))
    y2 = bias + phi * math.log(max(v - spread, _EPS))
    y_pred = _WM1_0 * y0 + _WM1_OTHER * (y1 + y2)
    p_yy = _WC1_0 * (y0 - y_pred) ** 2 + _WC1_OTHER * ((y1 - y_pred) ** 2 + (y2 - y_pred) ** 2) + r_noise
    # (v - v) = 0 for the center point, so p_vy only picks up the two
    # off-center points, at +-spread from v.
    p_vy = _WC1_OTHER * (spread * (y1 - y_pred) - spread * (y2 - y_pred))
    k = p_vy / p_yy
    v_post = v + k * (y_obs - y_pred)
    p_post = min(max(p - k * k * p_yy, _EPS), _P_CEIL)
    innov = y_obs - y_pred
    ll = -0.5 * (math.log(2 * math.pi * p_yy) + innov * innov / p_yy)
    return max(v_post, _EPS), p_post, ll


@njit(cache=True)
def _student_t_ll(z: float, nu: float) -> float:
    """log-density of a UNIT-VARIANCE Student-t(nu) at ``z``, exact (no
    moment-matched Gaussian approximation). Verified against
    ``scipy.stats.t`` to float64 precision during development.

        log f_Z(z) = lgamma((nu+1)/2) - lgamma(nu/2) - 0.5*log(pi)
                     - 0.5*log(nu-2) - (nu+1)/2 * log(1 + z^2/(nu-2))
    """
    return (
        math.lgamma((nu + 1.0) / 2.0)
        - math.lgamma(nu / 2.0)
        - 0.5 * math.log(math.pi)
        - 0.5 * math.log(nu - 2.0)
        - (nu + 1.0) / 2.0 * math.log(1.0 + z * z / (nu - 2.0))
    )


@njit(cache=True)
def _filter_nb(
    mu: float,
    kappa: float,
    theta_v: float,
    xi: float,
    zeta: float,
    su2: float,
    gamma: float,
    nu: float,
    phi: float,
    r: np.ndarray,
    log_rv: np.ndarray,
    v0: float,
    p0: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """JIT-compiled hot path: run the UKF over the whole window. A pure
    Python for-loop of scalar math over ~900-2600 days, re-evaluated
    hundreds of times per L-BFGS-B fit (one finite-difference gradient eval
    per free parameter, per iteration), was the actual cost driver --
    ~8.9s/window (~24h for the full walk-forward study) back when this had 6
    params and ran in plain Python, not the UKF math itself. Compiling this
    exact loop is what numba is for."""
    n = r.size
    v_pred_path = np.empty(n)
    v_post_path = np.empty(n)

    v = v0
    p = p0
    ll = 0.0
    lev = 0.0  # no lagged day for t=0
    for t in range(n):
        v_pred, p_pred = _predict(v, p, kappa, theta_v, xi, lev)
        v_pred_path[t] = v_pred

        # Return: exact quasi-likelihood contribution (density of r_t given
        # the PREDICTED, not yet updated, V), no state update -- see module
        # docstring for why the earlier Harvey-Ruiz-Shephard UKF update on
        # this channel was dropped.
        z = (r[t] - mu) / math.sqrt(v_pred)
        ll += _student_t_ll(z, nu) - 0.5 * math.log(v_pred)

        # RV: the one UKF state update.
        y_rv = log_rv[t]
        if math.isfinite(y_rv):
            v, p, ll2 = _update(v_pred, p_pred, y_rv, su2, zeta, phi)
            ll += ll2
        else:
            v, p = v_pred, p_pred
        v_post_path[t] = v

        # GJR-style lagged leverage kick for tomorrow's predict step: a down
        # day (r[t] < mu) raises V_{t+1} by gamma * (r[t]-mu)^2, on top of
        # mean reversion. Uses r[t], already observed by the time we move to
        # t+1 -- no lookahead.
        resid = r[t] - mu
        lev = gamma * resid * resid if resid < 0.0 else 0.0

    return v_pred_path, v_post_path, p, ll, lev


def _filter(
    theta_vec, r: np.ndarray, log_rv: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Python-facing wrapper: unpack the flat parameter vector scipy.optimize
    hands us and dispatch to the JIT-compiled ``_filter_nb``. Returns
    ``(v_pred path, v_post path, final posterior variance, total
    log-likelihood, the leverage kick for the *next* predict step)``."""
    mu, kappa, theta_v, xi, zeta, log_su2, gamma, nu, phi = theta_vec
    v0 = float(np.var(r))
    return _filter_nb(
        float(mu), float(kappa), float(theta_v), float(xi), float(zeta), math.exp(log_su2), float(gamma),
        float(nu), float(phi), r, log_rv, v0, v0 * v0,
    )


class RealizedSV:
    """Heston-style stochastic volatility (CIR variance, UKF quasi-MLE) with
    a GJR-style lagged leverage term, using the realized measure -- see
    module docstring."""

    name = "Realized-SV"

    def fit_predict(self, ctx: Context) -> PredictiveDist:
        r = np.asarray(ctx.returns, float)
        if not ctx.realized or "rv" not in ctx.realized or r.size < 120:
            return EmpiricalDist(r)
        rv = np.asarray(ctx.realized["rv"], float)
        rv = np.where(np.isfinite(rv) & (rv > 0), rv, np.nan)
        if np.isnan(rv).mean() > 0.2:
            return EmpiricalDist(r)
        log_rv = np.log(ffill(rv))[: r.size]

        var_r = float(np.var(r))
        vol_r = float(np.sqrt(var_r))
        # kappa bound: the Euler discretization's linearized map is
        # V_{t+1} ~= (1-kappa)*V_t + ..., dt=1 day -- stable only for
        # kappa < 2 (its own coefficient's absolute value < 1), same as any
        # AR(1) needing |coefficient| < 1. A bound/x0 above ~2 (e.g. kappa=5,
        # tried during development) makes the UKF's own state-variance
        # recursion diverge exponentially within ~20 days regardless of the
        # data, long before the optimizer can steer away from it.
        # gamma: GJR-style leverage strength on `resid^2` (variance units),
        # same role as GjrGarchT's own gamma -- typical fitted daily values
        # are O(0.05-0.3), so 0.1 is a modest, central starting guess.
        # nu: Student-t d.o.f. of the return innovation z_t, jointly fitted
        # (not a post-hoc fit on filtered residuals) since it's now baked
        # directly into the return's exact log-density contribution
        # (_student_t_ll) rather than a separate moment-matching step.
        # x0=8 matches RealizedGARCH's own nu fallback.
        # phi: RV measurement slope (log RV_t = zeta + phi*log V_t + u_t),
        # freed rather than fixed at 1 -- mirrors Realized-GARCH's own phi
        # (bounds (0.1, 3.0) there too), which lets RV's realized-vs-true
        # variance relationship scale non-1:1 (microstructure noise, diurnal
        # effects) instead of forcing the filter to reconcile a possibly
        # wrong slope only through zeta (an intercept can't fix a slope
        # mismatch).
        x0 = np.array([float(np.mean(r)), 0.05, var_r, 0.3 * vol_r, 0.0, np.log(0.3), 0.1, 8.0, 1.0])
        bounds = [
            (-0.05, 0.05),
            (1e-4, 1.9),
            (var_r * 1e-3, var_r * 20.0),
            (1e-8, 3.0 * vol_r),
            (-5.0, 5.0),
            (np.log(1e-4), np.log(5.0)),
            (0.0, 2.0),
            (3.0, 50.0),
            (0.1, 3.0),
        ]

        def nll(th):
            # numba's math.sqrt/log follow C semantics (nan on a domain
            # error, no Python exception), and _predict/_update already
            # clamp every sqrt/log argument to >= _EPS -- so a bad corner
            # shows up as a non-finite ll, not a raised exception.
            _, _, _, ll, _ = _filter(th, r, log_rv)
            return -ll if np.isfinite(ll) else 1e10

        try:
            res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 200})
            th = res.x if res.success else x0
        except Exception:  # noqa: BLE001
            return EmpiricalDist(r)

        mu, kappa, theta_v, xi, _zeta, _log_su2, _gamma, _nu, _phi = th
        v_pred_path, v_post_path, p_last, _, lev_next = _filter(th, r, log_rv)
        v_next, _ = _predict(v_post_path[-1], p_last, kappa, theta_v, xi, lev_next)
        sigma_next = float(np.sqrt(max(v_next, _EPS)))
        if not np.isfinite(sigma_next) or sigma_next <= 0 or sigma_next > 20 * r.std():
            return EmpiricalDist(r)

        # The parametric Student-t tail (fitted nu) gets the FZ0/MCS ranking
        # right (competitive, even best-in-class on BTC) but consistently
        # fails Kupiec/CC/DQ at 1% and the Acerbi-Szekely ES test at both
        # alphas -- a shape mismatch specifically in the extreme tail, not a
        # volatility-level problem (2.5% coverage already passes). A single
        # nu fit via MLE optimizes the *whole* likelihood, dominated by
        # ordinary days, not the extreme quantiles VaR/ES actually query.
        # GPD-POT on the filtered window's own standardized residuals fits
        # the tail directly instead of inferring it from a bulk-dominated nu
        # -- same McNeil & Frey (2000) construction GarchEVT already uses
        # (models/garch_evt.py), just on this model's own residual stream.
        z = (r - mu) / np.sqrt(np.maximum(v_pred_path, _EPS))
        return GpdTailDist(z, loc=mu, scale=sigma_next, threshold_q=0.90)
