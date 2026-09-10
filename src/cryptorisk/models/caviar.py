"""CAViaR - Conditional Autoregressive Value at Risk (Engle & Manganelli, 2004).

Models the conditional quantile directly, no distributional assumption. Specs:

* ``"SAV"``  q_t = b1 + b2 q_{t-1} + b3 |r_{t-1}|
* ``"AS"``   q_t = b1 + b2 q_{t-1} + b3 max(r_{t-1},0) + b4 max(-r_{t-1},0)

``CAViaR(..., exog=True)`` (a.k.a. CAViaR-X) adds ``b_x * sqrt(RV_{t-1})`` using
``ctx.realized["rv"]``. Parameters minimise the tick (pinball) loss; ES is the
in-sample mean of returns below the fitted quantile.

CAViaR is fit per ``alpha``, so the model is built with the study's alpha list
(``registry`` passes it from the config) and returns a
:class:`~cryptorisk.models.base.QuantileDist`.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter

from cryptorisk.models._util import ffill
from cryptorisk.models.base import Context, PredictiveDist, QuantileDist

_SPECS = {"SAV": 3, "AS": 4}


def _news(r: np.ndarray, spec: str) -> np.ndarray:
    """The exogenous 'news' term g(r_{t-1}) columns (without the AR feedback)."""
    if spec == "SAV":
        return np.abs(r)[:, None]
    return np.column_stack([np.maximum(r, 0.0), np.maximum(-r, 0.0)])


def _quantile_path(params: np.ndarray, r: np.ndarray, q_init: float, spec: str,
                   x: np.ndarray | None, news: np.ndarray) -> np.ndarray:
    """q_t = b0 + b1 q_{t-1} + (news @ b_news)_{t-1} [+ b_x x_{t-1}].

    An AR(1) with a time-varying intercept -> exact via ``lfilter``.
    """
    b0, b1 = params[0], params[1]
    n_news = news.shape[1]
    drive = b0 + news @ params[2 : 2 + n_news]
    if x is not None:
        drive = drive + params[-1] * x
    inp = np.empty(r.size)
    inp[0] = q_init
    inp[1:] = drive[:-1]                    # driven by info from t-1
    return lfilter([1.0], [1.0, -b1], inp, zi=[b1 * (q_init - inp[0])])[0]


def _step(b: np.ndarray, q_last: float, r_last: float, x_last: float | None, spec: str) -> float:
    if spec == "SAV":
        v = b[0] + b[1] * q_last + b[2] * abs(r_last)
    else:
        v = b[0] + b[1] * q_last + b[2] * max(r_last, 0.0) + b[3] * max(-r_last, 0.0)
    if x_last is not None:
        v += b[-1] * x_last
    return v


def _fit_one(r: np.ndarray, alpha: float, spec: str, x: np.ndarray | None):
    q_init = float(np.quantile(r[: min(300, r.size)], alpha))
    news = _news(r, spec)
    k = 2 + news.shape[1] + (1 if x is not None else 0)

    def loss(params):
        q = _quantile_path(params, r, q_init, spec, x, news)
        u = r - q
        return float(np.mean(u * (alpha - (u < 0).astype(float))))

    slope0 = [-0.2, -0.2] if spec == "AS" else [-0.2]
    starts = [
        np.array([q_init * 0.1, 0.9, *slope0, *([0.0] if x is not None else [])]),
        np.array([q_init * 0.02, 0.6, *[s * 2 for s in slope0], *([0.0] if x is not None else [])]),
    ]
    best, best_loss = None, np.inf
    for s in starts:
        try:
            res = minimize(loss, s[:k], method="Nelder-Mead",
                           options={"maxiter": 600, "xatol": 1e-6, "fatol": 1e-9})
            if np.isfinite(res.fun) and res.fun < best_loss:
                best, best_loss = res.x, res.fun
        except Exception:  # noqa: BLE001
            continue
    if best is None:
        return np.full(r.size, q_init), q_init
    q_path = _quantile_path(best, r, q_init, spec, x, news)
    q_next = _step(best, q_path[-1], r[-1], (x[-1] if x is not None else None), spec)
    return q_path, float(q_next)


class CAViaR:
    def __init__(self, alphas, spec: str = "AS", exog: bool = False, name: str | None = None):
        if spec not in _SPECS:
            raise ValueError(f"spec must be one of {list(_SPECS)}")
        self.alphas = tuple(alphas)
        self.spec = spec
        self.exog = exog
        self.name = name or (f"CAViaR-X-{spec}" if exog else f"CAViaR-{spec}")

    def fit_predict(self, ctx: Context) -> PredictiveDist:
        r = ctx.returns
        x = None
        if self.exog and ctx.realized and "rv" in ctx.realized:
            xv = np.asarray(ctx.realized["rv"], float)
            xv = np.where(np.isfinite(xv) & (xv > 0), xv, np.nan)
            if np.isnan(xv).mean() < 0.2:
                x = np.sqrt(ffill(xv))[: r.size]

        var_by_a, es_by_a = {}, {}
        for a in self.alphas:
            q_path, q_next = _fit_one(r, a, self.spec, x)
            v = float(q_next if q_next < 0 else -abs(q_next) - 1e-6)
            # ES from the in-sample ES/VaR ratio of the fitted quantile path,
            # so es scales with the one-step forecast and never sits above it.
            mask = r < q_path
            ratio = float(r[mask].mean() / q_path[mask].mean()) if mask.sum() >= 5 else 1.3
            ratio = min(max(ratio, 1.0), 3.0)
            var_by_a[a] = v
            es_by_a[a] = v * ratio

        # repair quantile crossing: a deeper tail (smaller alpha) must have a
        # more negative VaR/ES than a shallower one.
        for lo, hi in zip(sorted(self.alphas)[:-1], sorted(self.alphas)[1:], strict=True):
            var_by_a[lo] = min(var_by_a[lo], var_by_a[hi])
            es_by_a[lo] = min(es_by_a[lo], es_by_a[hi], var_by_a[lo])
        return QuantileDist(var_by_a, es_by_a)

