"""Perpetual-futures hedge analysis (V2_PLAN §6.4).

A short perpetual position of ratio ``h`` against a long spot position of
notional ``N`` has portfolio return ``r_spot - h * r_perp``. Two choices of
``h``:

* **minimum-variance**: ``h = Cov(spot, perp) / Var(perp)``;
* **ES-minimising**: the ``h`` that minimises the empirical ``alpha``-ES of the
  hedged return.

Carry is the perpetual funding on the hedged notional, annualised
(``mean(funding_8h) * 3 * 365``).  **Sign:** a positive funding rate means long
positions pay short positions, and the hedge is *short* the perp, so a positive
carry is *income* to the hedger.  Crypto perp funding has been predominantly
positive over 2019-2026, so this hedge has been a positive-carry trade.

The store has perp *funding* but no perp *price* series, so where a perp return
is unavailable the caller passes the spot return as a proxy (basis ~ 0): the
hedge ratios then sit near 1 and the informative output is the funding carry
against the capital freed. Basis risk needs a perp price feed and is reported
as unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar


def _empirical_es(x: np.ndarray, alpha: float) -> float:
    q = np.quantile(x, alpha)
    tail = x[x <= q]
    return float(tail.mean()) if tail.size else float(q)


def min_variance_ratio(spot_ret, perp_ret) -> float:
    s = np.asarray(spot_ret, float)
    p = np.asarray(perp_ret, float)
    ok = np.isfinite(s) & np.isfinite(p)
    s, p = s[ok], p[ok]
    c = np.cov(s, p)  # same ddof for numerator and denominator
    return float(c[0, 1] / c[1, 1]) if c[1, 1] > 0 else 0.0


def es_minimising_ratio(spot_ret, perp_ret, alpha: float, *, bounds=(-0.5, 2.5)) -> float:
    s = np.asarray(spot_ret, float)
    p = np.asarray(perp_ret, float)
    ok = np.isfinite(s) & np.isfinite(p)
    s, p = s[ok], p[ok]

    def neg_es(h: float) -> float:
        return -_empirical_es(s - h * p, alpha)  # ES is negative; minimise |ES|

    res = minimize_scalar(neg_es, bounds=bounds, method="bounded")
    return float(res.x)


def funding_carry_annualised(funding_8h) -> float:
    """Annualised perp funding rate: ``mean(funding_8h) * 3 * 365`` (funding is
    exchanged every 8h). Positive => longs pay shorts, so a short-perp hedge
    *earns* this."""
    f = np.asarray(funding_8h, float)
    f = f[np.isfinite(f)]
    return float(np.mean(f) * 3 * 365) if f.size else np.nan


@dataclass(frozen=True)
class HedgeSummary:
    ratio_min_var: float
    ratio_es_min: float
    es_unhedged: float  # alpha-ES of the spot return
    es_hedged: float  # alpha-ES at the ES-minimising ratio
    es_reduction: float  # 1 - es_hedged / es_unhedged
    funding_carry_annual_frac: float  # + => the short-perp hedge earns
    funding_carry_annual_usd: float
    note: str


def hedge_summary(
    spot_ret,
    perp_ret,
    funding_8h,
    *,
    alpha: float,
    notional: float,
    perp_is_proxy: bool,
) -> HedgeSummary:
    s = np.asarray(spot_ret, float)
    p = np.asarray(perp_ret, float)
    ok = np.isfinite(s) & np.isfinite(p)
    s, p = s[ok], p[ok]

    h_mv = min_variance_ratio(s, p)
    es_un = _empirical_es(s, alpha)
    carry = funding_carry_annualised(funding_8h)

    if perp_is_proxy:
        # perp == spot, so the ES-minimising ratio is degenerate (h -> 1 zeroes
        # the hedged series). Report only what is meaningful: the min-var ratio
        # (1.0), the unhedged ES and the funding carry on a 1x hedge.
        h_es = es_hg = es_red = np.nan
        hedge_units = abs(h_mv)
        note = "perp return proxied by spot (no perp price in store); ES-min hedge and ES reduction not meaningful, basis risk unavailable"
    else:
        h_es = es_minimising_ratio(s, p, alpha)
        es_hg = _empirical_es(s - h_es * p, alpha)
        es_red = float(1.0 - es_hg / es_un) if es_un != 0 else np.nan
        hedge_units = abs(h_es)
        note = ""

    return HedgeSummary(
        ratio_min_var=h_mv,
        ratio_es_min=h_es,
        es_unhedged=es_un,
        es_hedged=es_hg,
        es_reduction=es_red,
        funding_carry_annual_frac=carry,
        funding_carry_annual_usd=float(carry * hedge_units * notional)
        if np.isfinite(carry)
        else np.nan,
        note=note,
    )
