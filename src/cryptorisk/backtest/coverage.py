"""VaR coverage backtests (V2_PLAN §5.1).

* Kupiec unconditional coverage (POF) - Kupiec (1995)
* Christoffersen independence and conditional coverage - Christoffersen (1998)
* Basel traffic light - BCBS backtesting framework

Ported and hardened from v1 ``risk_models/backtest.py`` (which was itself
textbook-correct). The Engle-Manganelli Dynamic Quantile test lives in a
separate module and is added in Phase 3.

All functions take a boolean/0-1 array of violations
(``violation_t = realized_t < var_t``) and the tail probability ``alpha``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class TestResult:
    name: str
    statistic: float
    p_value: float
    df: int

    def rejects(self, level: float = 0.05) -> bool:
        """True if H0 is rejected (the model has a problem) at ``level``.

        A non-informative result (``p_value`` is NaN) does not reject.
        """
        return np.isfinite(self.p_value) and self.p_value < level

    def __repr__(self) -> str:
        verdict = "REJECT H0" if self.rejects() else "ok"
        return f"{self.name}: stat={self.statistic:.4f} p={self.p_value:.4f} -> {verdict}"


def _as_bits(violations) -> np.ndarray:
    v = np.asarray(violations)
    if v.dtype == bool:
        return v.astype(np.int8)
    v = v.astype(float)
    if not np.all(np.isin(v, (0.0, 1.0))):
        raise ValueError("violations must be boolean or 0/1")
    return v.astype(np.int8)


def kupiec_pof(violations, alpha: float) -> TestResult:
    """H0: the violation rate equals ``alpha``. LR ~ chi2(1)."""
    v = _as_bits(violations)
    n, x = v.size, int(v.sum())
    if x == 0 or x == n:
        return TestResult("Kupiec POF", np.nan, np.nan, 1)
    pi = x / n
    ll0 = (n - x) * np.log(1 - alpha) + x * np.log(alpha)
    ll1 = (n - x) * np.log(1 - pi) + x * np.log(pi)
    lr = -2.0 * (ll0 - ll1)
    return TestResult("Kupiec POF", float(lr), float(stats.chi2.sf(lr, 1)), 1)


def christoffersen_independence(violations) -> TestResult:
    """H0: violations are serially independent (no 1-lag Markov clustering).
    LR ~ chi2(1)."""
    v = _as_bits(violations)
    n00 = n01 = n10 = n11 = 0
    for a, b in zip(v[:-1], v[1:], strict=True):
        if a == 0 and b == 0:
            n00 += 1
        elif a == 0 and b == 1:
            n01 += 1
        elif a == 1 and b == 0:
            n10 += 1
        else:
            n11 += 1
    n0, n1 = n00 + n01, n10 + n11
    if n0 == 0 or n1 == 0:
        return TestResult("Christoffersen independence", np.nan, np.nan, 1)
    pi01 = n01 / n0
    pi11 = n11 / n1
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def slog(p: float) -> float:
        return np.log(p) if p > 0 else 0.0

    ll0 = (n00 + n10) * slog(1 - pi) + (n01 + n11) * slog(pi)
    ll1 = (
        n00 * slog(1 - pi01)
        + n01 * slog(pi01)
        + n10 * slog(1 - pi11)
        + n11 * slog(pi11)
    )
    lr = -2.0 * (ll0 - ll1)
    return TestResult("Christoffersen independence", float(lr), float(stats.chi2.sf(lr, 1)), 1)


def christoffersen_cc(violations, alpha: float) -> TestResult:
    """Conditional coverage: Kupiec POF + independence. LR ~ chi2(2)."""
    uc = kupiec_pof(violations, alpha)
    ind = christoffersen_independence(violations)
    if not (np.isfinite(uc.statistic) and np.isfinite(ind.statistic)):
        return TestResult("Christoffersen CC", np.nan, np.nan, 2)
    lr = uc.statistic + ind.statistic
    return TestResult("Christoffersen CC", float(lr), float(stats.chi2.sf(lr, 2)), 2)


# Basel traffic light: exceptions in a 250-day window -> capital multiplier add-on
# (BCBS 1996 backtesting framework). Green 0-4, Amber 5-9, Red >= 10.
_BASEL_ADDON = {4: 0.00, 5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}


@dataclass(frozen=True)
class BaselResult:
    exceptions: int
    window: int
    zone: str
    multiplier_addon: float


def basel_traffic_light(violations, window: int = 250) -> BaselResult:
    """Zone and capital multiplier add-on from the exception count over the
    last ``window`` days. Defined for a 99% VaR (alpha = 0.01)."""
    v = _as_bits(violations)
    x = int(v[-window:].sum()) if v.size >= window else int(v.sum())
    if x <= 4:
        zone, addon = "green", 0.0
    elif x <= 9:
        zone, addon = "amber", _BASEL_ADDON[x]
    else:
        zone, addon = "red", 1.0
    return BaselResult(x, min(window, v.size), zone, addon)


def run_all(violations, alpha: float) -> list[TestResult]:
    return [
        kupiec_pof(violations, alpha),
        christoffersen_independence(violations),
        christoffersen_cc(violations, alpha),
    ]
