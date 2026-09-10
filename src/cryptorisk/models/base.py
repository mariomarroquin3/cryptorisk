"""Model interface: :class:`Context` in, :class:`PredictiveDist` out.

Conventions (shared with the whole project):

* Work in **log-returns**.
* ``alpha`` is a **lower-tail probability** (0.025 -> the 97.5% VaR).
* ``var(alpha)`` and ``es(alpha)`` are returned as **return levels**, normally
  negative. A violation on day ``t`` is ``realized_return_t < var_t``.
* A model sees only information up to and including ``ctx.asof``; its forecast is
  for the next trading day.

Every model implements :class:`Model` (a ``Protocol``): ``name`` plus
``fit_predict(ctx) -> PredictiveDist``. The walk-forward engine calls
``fit_predict`` once per day and then queries the returned distribution at each
configured ``alpha``.

Three concrete distributions cover every model family:

* :class:`ParametricDist` - location/scale with a standardized reference
  distribution (GARCH family, EWMA, ...).
* :class:`EmpiricalDist` - quantiles from a (optionally weighted) sample
  (Historical Simulation, FHS, GARCH-EVT residual step).
* :class:`QuantileDist` - only specific (VaR, ES) pairs are known
  (CAViaR); no full density.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


# --------------------------------------------------------------------------- #
# Context
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Context:
    """Everything a model may use to produce one 1-step-ahead forecast.

    Parameters
    ----------
    returns
        The estimation window of log-returns, chronological, no NaN.
    dates
        ``datetime64`` array aligned 1:1 with ``returns``.
    asof
        Date of the last observation in ``returns``; the forecast is for the
        next trading day.
    asset
        Asset ticker (e.g. ``"BTC"``).
    realized
        Optional intraday-derived series aligned with ``returns``
        (``"rv"``, ``"bv"``, ``"rsv_pos"``, ``"rsv_neg"``, ``"jump"``).
    exog
        Optional exogenous series aligned with ``returns``
        (``"funding"``, ``"oi"``, ``"netflow"``, ...), for the ``-X`` models.
    """

    returns: np.ndarray
    dates: np.ndarray
    asof: np.datetime64
    asset: str = "BTC"
    realized: Mapping[str, np.ndarray] | None = None
    exog: Mapping[str, np.ndarray] | None = None

    def __post_init__(self) -> None:
        r = np.asarray(self.returns, dtype=float)
        if r.ndim != 1 or r.size < 2:
            raise ValueError("Context.returns must be 1-D with >= 2 observations")
        if not np.all(np.isfinite(r)):
            raise ValueError("Context.returns contains non-finite values")
        if len(self.dates) != len(r):
            raise ValueError("Context.dates and returns length mismatch")
        object.__setattr__(self, "returns", r)

    @property
    def n(self) -> int:
        return self.returns.size


# --------------------------------------------------------------------------- #
# Predictive distribution
# --------------------------------------------------------------------------- #
class PredictiveDist:
    """1-step-ahead predictive distribution of the log-return.

    Subclasses must provide :meth:`var` and :meth:`es`. :meth:`sigma2`,
    :meth:`cdf` and :meth:`ppf` are optional and raise
    :class:`NotImplementedError` when the model cannot supply them (that is a
    valid state, e.g. CAViaR has no density).
    """

    def var(self, alpha: float) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    def es(self, alpha: float) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    def sigma2(self) -> float:
        raise NotImplementedError(f"{type(self).__name__} does not expose sigma2")

    def cdf(self, x: float) -> float:
        raise NotImplementedError(f"{type(self).__name__} has no cdf")

    def ppf(self, u: float) -> float:
        raise NotImplementedError(f"{type(self).__name__} has no ppf")

    @staticmethod
    def _check_alpha(alpha: float) -> None:
        if not 0.0 < alpha < 0.5:
            raise ValueError(f"alpha must be a tail probability in (0, 0.5), got {alpha}")


@dataclass
class ParametricDist(PredictiveDist):
    """Location-scale distribution ``loc + scale * Z`` where ``Z`` has a fixed
    standardized reference distribution (unit-variance where applicable).

    Parameters
    ----------
    loc, scale
        Conditional mean and standard deviation of the log-return.
    z_ppf, z_cdf
        Quantile and CDF of the standardized innovation ``Z``.
    z_es
        Optional closed-form lower-tail ES of ``Z`` at level ``alpha``
        (E[Z | Z <= z_ppf(alpha)]). If absent it is computed by quadrature of
        ``z_ppf``.
    """

    loc: float
    scale: float
    z_ppf: Callable[[float], float]
    z_cdf: Callable[[float], float]
    z_es: Callable[[float], float] | None = None

    def __post_init__(self) -> None:
        if not (self.scale > 0 and math.isfinite(self.scale)):
            raise ValueError(f"scale must be positive and finite, got {self.scale}")

    def var(self, alpha: float) -> float:
        self._check_alpha(alpha)
        return self.loc + self.scale * float(self.z_ppf(alpha))

    def es(self, alpha: float) -> float:
        self._check_alpha(alpha)
        if self.z_es is not None:
            z_tail = float(self.z_es(alpha))
        else:
            grid = np.linspace(1e-6, alpha, 512)
            z_tail = float(np.trapezoid([self.z_ppf(u) for u in grid], grid) / alpha)
        return self.loc + self.scale * z_tail

    def sigma2(self) -> float:
        return self.scale**2

    def cdf(self, x: float) -> float:
        return float(self.z_cdf((x - self.loc) / self.scale))

    def ppf(self, u: float) -> float:
        return self.loc + self.scale * float(self.z_ppf(u))


@dataclass
class EmpiricalDist(PredictiveDist):
    """Quantiles from a sample, optionally weighted (weights sum to 1).

    Used for Historical Simulation, age-weighted HS, and the residual step of
    Filtered HS / GARCH-EVT (where the sample is standardized residuals and the
    caller rescales by ``sigma_{t+1}``).
    """

    sample: np.ndarray
    weights: np.ndarray | None = None
    scale: float = 1.0
    loc: float = 0.0
    sigma2_value: float | None = None
    _order: np.ndarray = field(init=False, repr=False)
    _cum: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        s = np.asarray(self.sample, dtype=float)
        s = s[np.isfinite(s)]
        if s.size < 2:
            raise ValueError("EmpiricalDist needs >= 2 finite sample points")
        if self.weights is None:
            w = np.full(s.size, 1.0 / s.size)
        else:
            w = np.asarray(self.weights, dtype=float)
            if w.shape != s.shape or np.any(w < 0):
                raise ValueError("weights must match sample and be non-negative")
            w = w / w.sum()
        order = np.argsort(s)
        object.__setattr__(self, "sample", s)
        object.__setattr__(self, "weights", w)
        object.__setattr__(self, "_order", order)
        object.__setattr__(self, "_cum", np.cumsum(w[order]))

    def _weighted_quantile(self, alpha: float) -> float:
        s_sorted = self.sample[self._order]
        idx = int(np.searchsorted(self._cum, alpha, side="left"))
        idx = min(idx, s_sorted.size - 1)
        return float(s_sorted[idx])

    def var(self, alpha: float) -> float:
        self._check_alpha(alpha)
        return self.loc + self.scale * self._weighted_quantile(alpha)

    def es(self, alpha: float) -> float:
        self._check_alpha(alpha)
        q = self._weighted_quantile(alpha)
        mask = self.sample <= q
        if not mask.any():
            tail = q
        else:
            w = self.weights[mask]
            tail = float(np.sum(self.sample[mask] * w) / w.sum())
        return self.loc + self.scale * tail

    def cdf(self, x: float) -> float:
        z = (x - self.loc) / self.scale
        return float(np.sum(self.weights[self.sample <= z]))

    def ppf(self, u: float) -> float:
        if not 0.0 < u < 1.0:
            raise ValueError("u must be in (0, 1)")
        return self.loc + self.scale * self._weighted_quantile(u)

    def sigma2(self) -> float:
        if self.sigma2_value is not None:
            return self.sigma2_value
        # fall back to the sample variance of loc + scale * z
        return float(self.scale**2 * np.var(self.sample))


@dataclass
class QuantileDist(PredictiveDist):
    """Only specific (VaR, ES) pairs are known (CAViaR). No density.

    ``var_by_alpha`` / ``es_by_alpha`` map an alpha to its level. Querying an
    alpha that was not pre-computed raises ``KeyError``.
    """

    var_by_alpha: Mapping[float, float]
    es_by_alpha: Mapping[float, float]
    _sigma2: float | None = None

    def var(self, alpha: float) -> float:
        self._check_alpha(alpha)
        return float(self.var_by_alpha[_key(self.var_by_alpha, alpha)])

    def es(self, alpha: float) -> float:
        self._check_alpha(alpha)
        return float(self.es_by_alpha[_key(self.es_by_alpha, alpha)])

    def sigma2(self) -> float:
        if self._sigma2 is None:
            raise NotImplementedError("QuantileDist has no sigma2")
        return self._sigma2


def _key(mapping: Mapping[float, float], alpha: float, tol: float = 1e-9) -> float:
    for k in mapping:
        if abs(k - alpha) <= tol:
            return k
    raise KeyError(f"alpha {alpha} not pre-computed; available: {sorted(mapping)}")


# --------------------------------------------------------------------------- #
# Model protocol
# --------------------------------------------------------------------------- #
@runtime_checkable
class Model(Protocol):
    """A risk model. Stateless across days: the engine calls ``fit_predict``
    fresh each step (models may cache internally keyed on the window)."""

    name: str

    def fit_predict(self, ctx: Context) -> PredictiveDist: ...
