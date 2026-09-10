"""Comparative scoring: consistent losses + Model Confidence Set (V2_PLAN §5.3).

* ``fz0_loss``  — Fissler–Ziegel 0-homogeneous joint (VaR, ES) loss, strictly
  consistent at level ``alpha`` (Patton, Ziegel & Chen, 2019). Lower is better.
* ``qlike_loss`` — QLIKE for a variance forecast vs a realized proxy
  (Patton, 2011); robust to a noisy proxy. Lower is better.
* ``diebold_mariano`` — paired equal-predictive-ability test on a loss diff,
  Newey–West HAC variance + Harvey–Leybourne–Newbold small-sample correction.
* ``model_confidence_set`` — Hansen, Lunde & Nason (2011), T_max variant with a
  stationary block bootstrap. Returns the set of models that are not
  significantly worse than the best, plus each model's MCS p-value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def fz0_loss(realized, var, es, alpha: float) -> np.ndarray:
    r"""Per-observation FZ0 loss for the (VaR, ES) pair at level ``alpha``.

    With :math:`v_t=\mathrm{VaR}_t<0`, :math:`e_t=\mathrm{ES}_t<0`:

    .. math::
        L^{FZ0}_t = \frac{1}{\alpha e_t}\,\mathbf 1\{r_t\le v_t\}(v_t-r_t)
                    + \frac{v_t}{e_t} + \ln(-e_t) - 1
    """
    r, v, e = (np.asarray(x, float) for x in (realized, var, es))
    e = np.where(e < -1e-12, e, -1e-12)  # ES must be strictly negative
    hit = (r <= v).astype(float)
    # breach term is positive: 1/(alpha*e) < 0 and (r - v) < 0 on a breach.
    return (1.0 / (alpha * e)) * hit * (r - v) + v / e + np.log(-e) - 1.0


def qlike_loss(sigma2_forecast, realized_var) -> np.ndarray:
    """QLIKE = rv/h - ln(rv/h) - 1 (>= 0, min at rv == h)."""
    h = np.asarray(sigma2_forecast, float)
    rv = np.asarray(realized_var, float)
    h = np.where(h > 1e-18, h, 1e-18)
    ratio = np.where(rv > 1e-18, rv, 1e-18) / h
    return ratio - np.log(ratio) - 1.0


# --------------------------------------------------------------------------- #
# Diebold–Mariano
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DMResult:
    statistic: float
    p_value: float  # two-sided
    mean_diff: float  # mean(loss_a - loss_b); < 0 -> A better

    def favors(self) -> str:
        return "A" if self.mean_diff < 0 else "B"


def _nw_var(d: np.ndarray, lag: int) -> float:
    d = d - d.mean()
    n = d.size
    g0 = float(d @ d / n)
    acc = g0
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1)
        gk = float(d[k:] @ d[:-k] / n)
        acc += 2.0 * w * gk
    return acc / n


def diebold_mariano(loss_a, loss_b, *, hac_lag: int | None = None) -> DMResult:
    a = np.asarray(loss_a, float)
    b = np.asarray(loss_b, float)
    if a.shape != b.shape:
        raise ValueError("loss series must be the same length")
    d = a - b
    n = d.size
    lag = hac_lag if hac_lag is not None else max(1, int(round(n ** (1 / 3))))
    var = _nw_var(d, lag)
    if var <= 0:
        return DMResult(np.nan, np.nan, float(d.mean()))
    dm = float(d.mean() / np.sqrt(var))
    # Harvey, Leybourne & Newbold (1997) small-sample correction
    corr = np.sqrt((n + 1 - 2 * lag + lag * (lag - 1) / n) / n)
    dm_hln = dm * corr
    p = 2.0 * float(stats.t.sf(abs(dm_hln), df=n - 1))
    return DMResult(dm_hln, p, float(d.mean()))


# --------------------------------------------------------------------------- #
# Giacomini–White conditional predictive ability (V2_PLAN §5.5)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GWResult:
    statistic: float
    p_value: float
    df: int
    beta: tuple[float, ...]  # OLS of d_t on h_{t-1}; beta[0] is the constant
    favors_high: str  # which model is better when the instrument is high

    def rejects(self, level: float = 0.05) -> bool:
        return np.isfinite(self.p_value) and self.p_value < level


def _nw_cov(z: np.ndarray, lag: int) -> np.ndarray:
    """Newey–West HAC estimate of the long-run variance of the rows of ``z``
    (shape ``(n, q)``): ``Gamma_0 + sum_k w_k (Gamma_k + Gamma_k')``. This is
    the asymptotic variance of ``sqrt(n) * mean(z)``."""
    z = z - z.mean(axis=0)
    n = z.shape[0]
    omega = z.T @ z / n
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1)
        gk = z[k:].T @ z[:-k] / n
        omega += w * (gk + gk.T)
    return omega


def giacomini_white(loss_a, loss_b, instruments, *, hac_lag: int | None = None) -> GWResult:
    """Giacomini & White (2006) test of equal *conditional* predictive ability.

    ``instruments`` is an ``(n, q)`` matrix of ``F_{t-1}``-measurable test
    functions (include a column of ones). With ``d_t = L^A_t - L^B_t`` and
    ``Z_t = h_{t-1} d_t``, under H0 ``E[Z_t] = 0`` and
    ``GW = n * Zbar' Omega^{-1} Zbar ~ chi2(q)`` (``Omega`` a HAC estimate).
    Rejection means the accuracy gap between A and B varies with the
    conditioning information (e.g. it depends on the volatility regime).
    """
    a = np.asarray(loss_a, float)
    b = np.asarray(loss_b, float)
    h = np.atleast_2d(np.asarray(instruments, float))
    if h.shape[0] != a.size:
        h = h.T
    d = a - b
    n, q = h.shape
    if n != d.size or n <= q + 2:
        return GWResult(np.nan, np.nan, q, tuple(np.full(q, np.nan)), "?")

    z = h * d[:, None]
    zbar = z.mean(axis=0)
    lag = hac_lag if hac_lag is not None else max(1, int(round(n ** (1 / 3))))
    omega = _nw_cov(z, lag)
    try:
        stat = float(n * zbar @ np.linalg.solve(omega, zbar))
    except np.linalg.LinAlgError:
        return GWResult(np.nan, np.nan, q, tuple(np.full(q, np.nan)), "?")

    beta, *_ = np.linalg.lstsq(h, d, rcond=None)  # direction of the difference
    # instrument in the last column is the regime signal; sign of its slope says
    # who wins when it is high (d = L_A - L_B < 0 -> A better)
    favors_high = "A" if beta[-1] < 0 else "B"
    return GWResult(stat, float(stats.chi2.sf(stat, q)), q, tuple(map(float, beta)), favors_high)


# --------------------------------------------------------------------------- #
# Model Confidence Set
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MCSResult:
    included: list[str]
    p_values: dict[str, float]
    best: str

    def __repr__(self) -> str:
        return f"MCS(best={self.best!r}, |set|={len(self.included)}: {self.included})"


def _stationary_bootstrap_indices(n: int, block_len: float, n_boot: int, rng) -> np.ndarray:
    p = 1.0 / max(block_len, 1.0)
    idx = np.empty((n_boot, n), dtype=np.int64)
    for b in range(n_boot):
        out = idx[b]
        cur = rng.integers(n)
        for t in range(n):
            if t > 0 and rng.random() < p:
                cur = rng.integers(n)
            out[t] = cur
            cur = (cur + 1) % n
    return idx


def model_confidence_set(
    losses: dict[str, np.ndarray],
    *,
    confidence: float = 0.90,
    block_len: int = 20,
    n_boot: int = 5000,
    seed: int | None = None,
) -> MCSResult:
    """T_max MCS. ``losses``: model name -> per-obs loss array (same length)."""
    names = list(losses)
    L = np.vstack([np.asarray(losses[k], float) for k in names])  # (m, T)
    m, T = L.shape
    if m < 2:
        return MCSResult(names, dict.fromkeys(names, 1.0), names[0])

    rng = np.random.default_rng(seed)
    boot = _stationary_bootstrap_indices(T, block_len, n_boot, rng)
    alpha = 1.0 - confidence

    alive = list(range(m))
    p_running = 0.0
    pvals: dict[str, float] = {}

    while len(alive) > 1:
        sub = L[alive]  # (k, T)
        d = sub - sub.mean(axis=0)  # excess loss vs the set mean
        dbar = d.mean(axis=1)  # (k,)
        db = d[:, boot].mean(axis=2)  # (k, n_boot) bootstrapped means
        v = np.maximum(db.var(axis=1, ddof=1), 1e-30)
        t_stat = dbar / np.sqrt(v)
        T_max = float(t_stat.max())
        T_max_boot = ((db - dbar[:, None]) / np.sqrt(v)[:, None]).max(axis=0)
        p = float((T_max_boot >= T_max).mean())

        p_running = max(p_running, p)
        if p_running >= alpha:  # equal predictive ability -> stop
            break
        worst = alive[int(np.argmax(t_stat))]
        pvals[names[worst]] = p_running
        alive.remove(worst)

    for i in alive:  # everything still standing is in the MCS
        pvals.setdefault(names[i], max(p_running, alpha))
    for k in names:
        pvals.setdefault(k, 1.0)

    included = sorted((names[i] for i in alive), key=lambda k: float(np.mean(losses[k])))
    best = min(names, key=lambda k: float(np.mean(losses[k])))
    return MCSResult(included, pvals, best)
