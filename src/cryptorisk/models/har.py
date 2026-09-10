"""HAR-RV and HARQ (V2_PLAN §3).

* ``HAR()``  - Corsi (2009): RV_{t+1} regressed on the daily, weekly (5d) and
  monthly (22d) averages of past RV, in levels, by OLS.
* ``HAR(harq=True)`` - Bollerslev, Patton & Quaedvlieg (2016): the daily lag is
  interacted with sqrt(realized quarticity) to down-weight noisy days.

Both need ``ctx.realized["rv"]`` (and ``["rq"]`` for HARQ). Without it they fall
back to the empirical quantile of the window. The RV forecast IS the one-step
conditional variance; the return distribution is a standardized Student-t whose
d.o.f. is fitted to r_t / sqrt(RV_t).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from cryptorisk.models._dist import student_t_z
from cryptorisk.models._util import ffill
from cryptorisk.models.base import Context, EmpiricalDist, ParametricDist, PredictiveDist

_W, _M = 5, 22


def _har_regressors(rv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Design matrix rows for t = _M .. n-1 (predicting rv[t] from info < t),
    plus the single row for the one-step forecast (uses rv up to n-1)."""
    n = rv.size
    cs = np.concatenate([[0.0], np.cumsum(rv)])

    def avg(t, k):  # mean of rv[t-k : t]
        return (cs[t] - cs[t - k]) / k

    rows, y = [], []
    for t in range(_M, n):
        rows.append([1.0, rv[t - 1], avg(t, _W), avg(t, _M)])
        y.append(rv[t])
    fcast = np.array([1.0, rv[n - 1], avg(n, _W), avg(n, _M)])
    return np.array(rows), np.array(y), fcast


class HAR:
    def __init__(self, harq: bool = False, name: str | None = None):
        self.harq = harq
        self.name = name or ("HARQ" if harq else "HAR-RV")

    def fit_predict(self, ctx: Context) -> PredictiveDist:
        rlz = ctx.realized
        if rlz is None or "rv" not in rlz:
            return EmpiricalDist(ctx.returns)
        rv = np.asarray(rlz["rv"], float)
        rv = np.where(np.isfinite(rv) & (rv > 0), rv, np.nan)
        if np.isnan(rv).mean() > 0.2 or rv.size < _M + 60:
            return EmpiricalDist(ctx.returns)
        rv = ffill(rv)

        X, y, xf = _har_regressors(rv)

        if self.harq and "rq" in rlz:
            rq = ffill(np.where(np.isfinite(rlz["rq"]) & (rlz["rq"] > 0), rlz["rq"], np.nan))
            sq = np.sqrt(rq)
            # z-score sqrt(RQ) so the interaction term sits on the same scale as
            # RV_d/RV_w/RV_m (raw sqrt(RQ) is ~500x smaller -> OLS coef -> 0).
            sq_z = (sq - np.nanmean(sq)) / (np.nanstd(sq) + 1e-30)
            extra = np.array([sq_z[t - 1] * rv[t - 1] for t in range(_M, rv.size)])
            X = np.column_stack([X, extra])
            xf = np.append(xf, sq_z[-1] * rv[-1])

        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        h_next = max(float(xf @ beta), np.nanpercentile(rv, 5) * 0.5, 1e-8)

        # innovation d.o.f. from standardized returns r_t / sqrt(RV_t)
        z = ctx.returns / np.sqrt(rv[: ctx.returns.size])
        z = z[np.isfinite(z)]
        nu = float(np.clip(stats.t.fit(z, floc=0)[0], 3.0, 50.0)) if z.size > 50 else 6.0
        ppf, cdf, es = student_t_z(nu)
        return ParametricDist(loc=0.0, scale=float(np.sqrt(h_next)), z_ppf=ppf, z_cdf=cdf, z_es=es)

