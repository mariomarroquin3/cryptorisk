"""P&L attribution and the FRTB PLA test (V2_PLAN §6.3).

The FRTB profit-and-loss attribution test compares two P&L series over a
window: the risk-theoretical P&L (**RTPL**, from the risk model) and the
hypothetical P&L (**HPL**, the desk's actual). Two metrics:

* **Spearman** rank correlation of RTPL vs HPL -- does the model preserve the
  ordering of daily outcomes;
* **Kolmogorov-Smirnov** distance between the two empirical distributions --
  does it get the *shape* right.

FRTB-style zones (Basel 2019 thresholds): green if
``spearman >= 0.80 and ks <= 0.09``; amber if ``spearman >= 0.70 and
ks <= 0.12``; red otherwise.

Here the single risk factor is the asset return, so we define
``RTPL_t = N * sigma_t * Phi^{-1}(PIT_t)`` -- the P&L the model implies once the
realized outcome is mapped through its own predictive CDF -- against
``HPL_t = N * r_t``. A well-calibrated model has ``RTPL_t ~= HPL_t``; a model
whose predictive scale or shape is wrong shows up as a large KS.

:func:`attribute_pnl` gives the drift / diffusion / jump / residual split of a
model's mean-path P&L for the descriptive table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class PLAResult:
    spearman: float
    ks: float
    zone: str


def pla_test(rtpl, hpl) -> PLAResult:
    a = np.asarray(rtpl, float)
    b = np.asarray(hpl, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 20:
        return PLAResult(np.nan, np.nan, "red")
    sp = float(stats.spearmanr(a, b).statistic)
    ks = float(stats.ks_2samp(a, b, method="asymp").statistic)
    if sp >= 0.80 and ks <= 0.09:
        zone = "green"
    elif sp >= 0.70 and ks <= 0.12:
        zone = "amber"
    else:
        zone = "red"
    return PLAResult(sp, ks, zone)


def implied_rtpl(sigma2, pit, notional: float) -> np.ndarray:
    """``RTPL_t = N * sigma_t * Phi^{-1}(PIT_t)`` (see module docstring)."""
    s = np.sqrt(np.asarray(sigma2, float))
    u = np.clip(np.asarray(pit, float), 1e-6, 1 - 1e-6)
    return notional * s * stats.norm.ppf(u)


@dataclass(frozen=True)
class PnlAttribution:
    drift: float
    diffusion: float
    jump: float
    residual: float


def attribute_pnl(
    realized_returns,
    mu,
    sigma2,
    notional: float,
    *,
    jump_var_share: float = 0.0,
) -> PnlAttribution:
    """Variance-based split of the ``N``-notional P&L over the window.

    ``drift`` is ``N * mean(mu)``; the return variance ``N^2 * var(r)`` is
    split into a jump part (``jump_var_share`` of it, non-zero only for the
    jump-diffusion model) and a diffusion part (the rest, capped at the model's
    mean ``sigma2``); ``residual`` is whatever variance the model did not
    account for. All figures are standard deviations in currency units.
    """
    r = np.asarray(realized_returns, float)
    r = r[np.isfinite(r)]
    n2 = notional**2
    total_var = n2 * float(np.var(r))
    drift = notional * float(np.mean(np.asarray(mu, float)))
    jump_var = jump_var_share * total_var
    model_var = n2 * float(np.mean(np.asarray(sigma2, float)))
    diffusion_var = max(model_var - jump_var, 0.0)
    residual_var = max(total_var - diffusion_var - jump_var, 0.0)
    return PnlAttribution(
        drift=float(drift),
        diffusion=float(np.sqrt(diffusion_var)),
        jump=float(np.sqrt(jump_var)),
        residual=float(np.sqrt(residual_var)),
    )
