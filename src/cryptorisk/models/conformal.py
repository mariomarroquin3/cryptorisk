"""Adaptive conformal recalibration of any model's VaR/ES (ACI, Gibbs & Candes 2021).

A miscalibrated model can be repaired after the fact without touching it: track how
often its VaR was actually breached and shift the *level* it is queried at. If the
model is breached too often, ask it for a deeper quantile tomorrow; if it is breached
too rarely, a shallower one. The update is the ACI recursion

    level_{t+1} = level_t + gamma * (alpha - err_t),      err_t = 1{r_t < VaR_t}

which gives long-run coverage ``alpha`` whatever the model does, with no
distributional assumption and no look-ahead (``err_t`` is known before day ``t+1`` is
forecast). The wrapper reuses the base model's fit and only changes the level at which
its predictive distribution is read, so VaR and ES stay consistent with each other.

It is *not* registered in ``all_models()``: the frozen study compares the raw models, and
``study.run_conformal`` evaluates the wrapped ones next to them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cryptorisk.models.base import Context, Model, PredictiveDist


class AdjustedDist(PredictiveDist):
    """``base`` read at a recalibrated tail level instead of the requested one."""

    def __init__(self, base: PredictiveDist, levels: dict[float, float]):
        self.base = base
        self._levels = {round(a, 10): lv for a, lv in levels.items()}

    def level(self, alpha: float) -> float:
        return self._levels.get(round(alpha, 10), alpha)

    def var(self, alpha: float) -> float:
        return self.base.var(self.level(alpha))

    def es(self, alpha: float) -> float:
        return self.base.es(self.level(alpha))

    def sigma2(self) -> float:
        return self.base.sigma2()

    def cdf(self, x: float) -> float:
        return self.base.cdf(x)

    def ppf(self, u: float) -> float:
        return self.base.ppf(u)


@dataclass
class _AssetState:
    levels: dict[float, float]
    last_asof: np.datetime64 | None = None
    last_var: dict[float, float] = field(default_factory=dict)


class AdaptiveConformal:
    """Wrap ``base`` so its VaR/ES are recalibrated online with ACI.

    ``gamma_frac`` sets the step as a fraction of the target level (``gamma = gamma_frac *
    alpha``): 0.05 is the ratio of the published gamma=0.005 to a 10% level, and keeps the
    adaptation equally gentle at 2.5% and 1%. The adjusted level is clipped to
    ``[alpha / 20, 0.2]``. State is kept per asset and is meant for the sequential
    walk-forward: it is idempotent for a repeated ``asof`` and skips the update across gaps.
    """

    def __init__(
        self,
        base: Model,
        alphas: tuple[float, ...],
        *,
        gamma_frac: float = 0.05,
        name: str | None = None,
    ):
        self.base = base
        self.alphas = tuple(alphas)
        self.gamma_frac = gamma_frac
        self.name = name or f"{base.name}+ACI"
        self._state: dict[str, _AssetState] = {}
        #: (asof, asset, alpha, level) for every forecast issued, for plotting the drift
        self.trace: list[tuple[np.datetime64, str, float, float]] = []

    def _update(self, st: _AssetState, ctx: Context) -> None:
        """Apply yesterday's outcome, if the previous forecast was for the day just realized."""
        if st.last_asof is None or ctx.asof <= st.last_asof:
            return
        if ctx.asof - st.last_asof != np.timedelta64(1, "D"):
            return  # a gap: the last forecast does not correspond to ctx.returns[-1]
        realized = float(ctx.returns[-1])
        for a in self.alphas:
            err = 1.0 if realized < st.last_var[a] else 0.0
            step = self.gamma_frac * a * (a - err)
            st.levels[a] = float(np.clip(st.levels[a] + step, a / 20.0, 0.2))

    def fit_predict(self, ctx: Context) -> PredictiveDist:
        st = self._state.setdefault(ctx.asset, _AssetState(levels={a: a for a in self.alphas}))
        self._update(st, ctx)
        dist = self.base.fit_predict(ctx)
        adjusted = AdjustedDist(dist, dict(st.levels))
        st.last_asof = ctx.asof
        st.last_var = {a: float(adjusted.var(a)) for a in self.alphas}
        for a in self.alphas:
            self.trace.append((ctx.asof, ctx.asset, a, st.levels[a]))
        return adjusted
