"""Multi-day forward risk cone for the API's Overview chart (see
``app.py``'s ``/forecast/{asset}``, field ``cone``).

This is deliberately **not** part of the headline model comparison (FZ0/MCS,
``data/results/eval_*``) -- it's a presentation layer that extends two of the
study's specialized models to a longer, honestly-labeled horizon:

* **Jump-Diffusion**: the Merton mixture compounds *exactly* over H days --
  jump arrivals are i.i.d. Poisson, so H days means ``N ~ Poisson(H*lambda)``
  jumps and diffusion variance ``H*sigma**2``. Not an approximation.
* **GARCH-EVT**: the H-day cumulative variance is the *sum* of ``arch``'s
  per-step variance forecasts (a mean-reverting term structure), not a flat
  ``H*sigma**2`` -- reuses the same fitted GPD tail shape
  (:class:`~cryptorisk.models._gpd.GpdTailDist`), just rescaled to the H-day
  loc/scale. Still an approximation (it assumes the 1-day standardized tail
  shape carries over to the H-day aggregate), but a materially better one
  than naive sqrt(H) scaling of a single day's quantile.

**MS-GARCH is deliberately not folded into this time-indexed cone**: its
walk-forward regime signal has ~no out-of-sample predictive power (see
``models.msgarch_bridge``'s docstring -- ``prob_crisis_pred`` correlates
~0.07 with realized vol) and it can't be re-fit outside R anyway, so there is
no honest way to draw "the regime at day H" as a forecast line. Instead
:func:`regime_summary` exposes what a normal-regime day and a crisis-regime
day *statistically look like* -- the full-sample-fitted stationary vol and
Student-t tail shape per regime (``data/results/msgarch_regime_params.csv``,
written by ``msgarch/fit_msgarch_walkforward.R::regime_params`` from the same
fit already computed for ``prob_crisis_insample``) -- as a **distribution
comparison**, not a horizon-indexed prediction. The crisis regime's fitted
GARCH persistence sits close enough to the unit-root boundary that its raw
stationary variance can blow up numerically (v1 hit the identical pathology
-- see its CLAUDE.md's crisis/normal vol ratio cap); :func:`_clipped_crisis_vol`
caps the crisis/normal stationary-vol ratio to ``[1.4, 3.0]`` for the same
reason, rather than pass through a nonsensical scenario.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from cryptorisk.config import repo_root
from cryptorisk.models._gpd import GpdTailDist
from cryptorisk.models.base import EmpiricalDist, PredictiveDist
from cryptorisk.models.jump import estimate_jump_params

try:
    from arch import arch_model

    _ARCH = True
except ImportError:  # pragma: no cover
    _ARCH = False

_N_SIM = 20_000
_EVT_THRESHOLD_Q = 0.90


def jump_diffusion_cone(
    returns: np.ndarray, horizons: list[int], *, seed: int = 0, n_sim: int = _N_SIM
) -> dict[int, EmpiricalDist]:
    """H-day Merton jump-diffusion distributions, one per horizon in
    ``horizons`` -- an exact compounding (Poisson jump *count* scales with H,
    diffusion *variance* scales with H), not a sqrt(H) approximation."""
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    jp = estimate_jump_params(r)
    mu, sigma = float(r.mean()), float(r.std())
    k_bar = jp.k_bar()
    rng = np.random.default_rng(seed)
    out: dict[int, EmpiricalDist] = {}
    for h in horizons:
        eps = rng.standard_normal(n_sim)
        n_jumps = rng.poisson(jp.lam * h, size=n_sim)
        jump = np.where(
            n_jumps > 0,
            rng.normal(jp.mu_j * n_jumps, jp.sig_j * np.sqrt(np.maximum(n_jumps, 1))),
            0.0,
        )
        sim = h * (mu - jp.lam * k_bar) + sigma * np.sqrt(h) * eps + jump
        out[h] = EmpiricalDist(sample=sim)
    return out


def garch_evt_cone(
    returns: np.ndarray, horizons: list[int], *, threshold_q: float = _EVT_THRESHOLD_Q
) -> dict[int, PredictiveDist | None]:
    """H-day GARCH-EVT distributions: cumulative variance is the sum of
    ``arch``'s per-step forecasts (mean-reverting), the tail shape is the
    same fitted GPD reused at each horizon's rescaled loc/scale. ``None`` for
    a horizon if the GARCH fit or its forecast isn't available."""
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    max_h = max(horizons)
    if not _ARCH or r.size < 150:
        return dict.fromkeys(horizons)
    try:
        res = arch_model(r * 100.0, mean="Constant", vol="GARCH", p=1, q=1, dist="t", rescale=False).fit(
            disp="off", show_warning=False
        )
        z = np.asarray(res.std_resid, float)
        z = z[np.isfinite(z)]
        fc = res.forecast(horizon=max_h, reindex=False)
        step_var = np.asarray(fc.variance.iloc[-1], float) / (100.0**2)
        mu_next = float(res.params.get("mu", 0.0)) / 100.0
        if z.size < 150 or not np.isfinite(step_var).all():
            return dict.fromkeys(horizons)
    except Exception:  # noqa: BLE001
        return dict.fromkeys(horizons)

    out: dict[int, PredictiveDist | None] = {}
    for h in horizons:
        cum_var = float(np.sum(step_var[:h]))
        if not np.isfinite(cum_var) or cum_var <= 0:
            out[h] = None
            continue
        out[h] = GpdTailDist(z, loc=mu_next * h, scale=float(np.sqrt(cum_var)), threshold_q=threshold_q)
    return out


@lru_cache(maxsize=1)
def _regime_params() -> pd.DataFrame:
    path = repo_root() / "data" / "results" / "msgarch_regime_params.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _regime_rows(asset: str) -> tuple[pd.Series, pd.Series] | None:
    df = _regime_params()
    if df.empty:
        return None
    crisis = df[(df["asset"] == asset) & (df["regime"] == "crisis")]
    normal = df[(df["asset"] == asset) & (df["regime"] == "normal")]
    if crisis.empty or normal.empty:
        return None
    return normal.iloc[0], crisis.iloc[0]


def _clipped_crisis_vol(normal_vol: float, crisis_vol_raw: float) -> float | None:
    """The crisis regime's GARCH persistence (alpha1+beta) sits close enough
    to the unit-root boundary that 1/(1-alpha1-beta) -- and so the
    "stationary" variance -- can blow up numerically (v1 hit the identical
    pathology, see its CLAUDE.md: "ratio crisis/normal acotado a
    [1.4, 3.0]"). Cap the crisis/normal vol ratio the same way rather than
    pass through a nonsensical scenario (e.g. 75x normal vol)."""
    if not (np.isfinite(normal_vol) and normal_vol > 0):
        return None
    if not (np.isfinite(crisis_vol_raw) and crisis_vol_raw > 0):
        return None
    ratio = np.clip(crisis_vol_raw / normal_vol, 1.4, 3.0)
    return normal_vol * float(ratio)


def regime_summary(asset: str) -> dict[str, dict[str, float]] | None:
    """Raw (clipped) regime params for the two-panel "what does a normal day
    vs. a crisis day look like" distribution comparison -- mean is always 0
    (the MS-GARCH spec here has no drift term), so only vol (std) and the
    Student-t degrees of freedom (``nu``, tail shape) vary by regime."""
    rows = _regime_rows(asset)
    if rows is None:
        return None
    nrow, crow = rows
    normal_vol, nu_n, p_n = float(nrow["stat_vol"]), float(nrow["nu"]), float(nrow["p_stay"])
    crisis_vol = _clipped_crisis_vol(normal_vol, float(crow["stat_vol"]))
    nu_c, p_c = float(crow["nu"]), float(crow["p_stay"])
    if crisis_vol is None or not (np.isfinite(nu_n) and nu_n > 2.0 and np.isfinite(nu_c) and nu_c > 2.0):
        return None
    return {
        "normal": {"vol": normal_vol, "nu": nu_n, "p_stay": p_n if np.isfinite(p_n) else None},
        "crisis": {"vol": crisis_vol, "nu": nu_c, "p_stay": p_c if np.isfinite(p_c) else None},
    }
