"""Copula portfolio VaR / ES.

For a fixed-weight basket of ``k`` assets: filter each asset's volatility
(``marginal``), fit a ``k``-dimensional copula to the standardized-residual
pseudo-observations, simulate joint standardized shocks, scale by each asset's
one-step sigma and aggregate.

Residual inversion is FHS-style (empirical quantile of the standardized
residuals), so the marginals are semiparametric and only the *dependence* is
parametric. Families: ``independence``, ``gaussian`` (full correlation matrix),
``student_t`` (full correlation matrix, fixed df), ``clayton`` (exchangeable,
lower-tail dependence).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata
from statsmodels.distributions.copula.api import (
    ClaytonCopula,
    GaussianCopula,
    IndependenceCopula,
    StudentTCopula,
)

from cryptorisk.portfolio.marginal import MarginalFit, fit_marginal

_FAMILIES = ("independence", "gaussian", "student_t", "clayton")


def _pseudo_obs(z: np.ndarray) -> np.ndarray:
    n = z.size
    return (rankdata(z) - 0.5) / n


def _nearest_psd(c: np.ndarray) -> np.ndarray:
    """Clip eigenvalues to a small positive floor and renormalise to unit
    diagonal -- the estimated pseudo-correlation matrix can be indefinite for
    ``k > 2``."""
    c = (c + c.T) / 2.0
    w, v = np.linalg.eigh(c)
    w = np.clip(w, 1e-8, None)
    a = (v * w) @ v.T
    d = np.sqrt(np.clip(np.diag(a), 1e-18, None))
    a = a / np.outer(d, d)
    np.fill_diagonal(a, 1.0)
    return a


def _corr_from_param(param, k_dim: int) -> np.ndarray:
    """Coerce ``fit_corr_param`` output to a ``(k, k)`` PSD correlation matrix.

    statsmodels returns a scalar equicorrelation for ``k_dim == 2`` and a full
    matrix for ``k_dim >= 3``; a vech vector is also accepted defensively.
    """
    p = np.asarray(param, float)
    if p.ndim == 2 and p.shape == (k_dim, k_dim):
        c = p.copy()
    elif p.size == 1:
        c = np.full((k_dim, k_dim), float(p))
    elif p.ndim == 1 and p.size == k_dim * (k_dim - 1) // 2:
        c = np.eye(k_dim)
        iu = np.triu_indices(k_dim, 1)
        c[iu] = p
        c[(iu[1], iu[0])] = p
    else:  # pragma: no cover - unexpected shape, degrade to identity
        return np.eye(k_dim)
    np.fill_diagonal(c, 1.0)
    c = np.clip(c, -0.999, 0.999)
    np.fill_diagonal(c, 1.0)
    return _nearest_psd(c)


def _fit_copula(u: np.ndarray, family: str, *, t_df: float, k_dim: int = 2):
    """Return (family_used, params) where params feeds :func:`_sample_copula`.

    Any fit failure (or a degenerate parameter) falls back to the independence
    copula, so a bad window never crashes the walk-forward. ``params`` always
    carries ``k_dim`` so the sampler needs no extra context.
    """
    if family == "independence":
        return "independence", {"k_dim": k_dim}
    try:
        if family == "gaussian":
            c = _corr_from_param(GaussianCopula(k_dim=k_dim).fit_corr_param(u), k_dim)
            return "gaussian", {"corr": c, "k_dim": k_dim}
        if family == "student_t":
            c = _corr_from_param(StudentTCopula(k_dim=k_dim).fit_corr_param(u), k_dim)
            return "student_t", {"corr": c, "df": float(t_df), "k_dim": k_dim}
        if family == "clayton":
            theta = float(ClaytonCopula(k_dim=k_dim).fit_corr_param(u))
            if np.isfinite(theta) and theta > 1e-4:
                return "clayton", {"theta": float(np.clip(theta, 1e-4, 50.0)), "k_dim": k_dim}
            return "independence", {"k_dim": k_dim}   # no lower-tail dependence to model
    except Exception:  # noqa: BLE001 - statsmodels optimiser failure -> independence
        return "independence", {"k_dim": k_dim}
    raise ValueError(f"unknown copula family {family!r}")


def _sample_copula(family: str, params: dict, n: int, rng) -> np.ndarray:
    k = int(params.get("k_dim", 2))
    if family == "independence":
        return IndependenceCopula(k_dim=k).rvs(n, rng=rng)
    if family == "gaussian":
        return GaussianCopula(corr=params["corr"], k_dim=k).rvs(n, rng=rng)
    if family == "student_t":
        return StudentTCopula(corr=params["corr"], df=params["df"], k_dim=k).rvs(n, rng=rng)
    if family == "clayton":
        return ClaytonCopula(theta=params["theta"], k_dim=k).rvs(n, rng=rng)
    raise ValueError(family)


@dataclass
class PortfolioForecast:
    var: dict[float, float]
    es: dict[float, float]
    sigma2: float
    sim: np.ndarray            # simulated 1-day portfolio log-returns
    family_used: str
    params: dict


class CopulaVaR:
    """A walk-forward-able portfolio model. ``assets`` is the column order."""

    def __init__(
        self,
        family: str,
        assets: list[str],
        weights: dict[str, float],
        *,
        n_sim: int = 20_000,
        t_df: float = 5.0,
        seed: int = 0,
    ):
        if family not in _FAMILIES:
            raise ValueError(f"family must be one of {_FAMILIES}")
        self.family = family
        self.assets = list(assets)
        w = np.array([weights[a] for a in self.assets], float)
        self.w = w / w.sum()
        self.n_sim = int(n_sim)
        self.t_df = float(t_df)
        self.seed = int(seed)
        self.name = f"Copula-{family}"

    def fit_predict(
        self,
        window: dict[str, np.ndarray],
        alphas: list[float],
        *,
        marginals: dict[str, MarginalFit] | None = None,
    ) -> PortfolioForecast:
        fits = marginals or {a: fit_marginal(window[a]) for a in self.assets}
        z = [fits[a].z_resid for a in self.assets]

        # Every asset's z_resid must be the *same length* for _pseudo_obs to
        # pair them day-for-day -- window[a] are equal-length, positionally
        # aligned slices of a jointly-dropna'd basket (run_portfolio.
        # _copula_walk_forward), and neither the `arch` GARCH(1,1) fit nor the
        # EWMA fallback drops observations, so lengths always match in
        # practice. If they ever don't, truncating each to the last L points
        # (as before) would silently pair residuals from different calendar
        # days across assets -- degrade to independence instead of guessing.
        aligned = len({x.size for x in z}) == 1
        if not aligned:
            lmin = min(x.size for x in z)
            z = [x[-lmin:] for x in z]
        u = np.column_stack([_pseudo_obs(x) for x in z])

        family = self.family if aligned else "independence"
        fam, params = _fit_copula(u, family, t_df=self.t_df, k_dim=len(self.assets))
        rng = np.random.default_rng(self.seed)
        u_sim = np.clip(_sample_copula(fam, params, self.n_sim, rng), 1e-9, 1 - 1e-9)

        # FHS inverse: empirical quantile of each asset's standardized residuals
        shocks = np.column_stack(
            [np.quantile(z[i], u_sim[:, i]) for i in range(len(self.assets))]
        )
        mu = np.array([fits[a].mu_next for a in self.assets])
        sig = np.array([fits[a].sigma_next for a in self.assets])
        asset_ret = mu + shocks * sig               # (n_sim, k)
        port = asset_ret @ self.w                    # (n_sim,)

        var, es = {}, {}
        for al in alphas:
            q = float(np.quantile(port, al))
            tail = port[port <= q]
            var[al] = q
            es[al] = float(tail.mean()) if tail.size else q
        return PortfolioForecast(
            var=var, es=es, sigma2=float(np.var(port)), sim=port,
            family_used=fam, params=params,
        )
