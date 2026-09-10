"""Copula portfolio VaR / ES.

For a fixed-weight basket: filter each asset's volatility (``marginal``), fit a
2-D copula to the standardized-residual pseudo-observations, simulate joint
standardized shocks, scale by each asset's one-step sigma and aggregate.

Residual inversion is FHS-style (empirical quantile of the standardized
residuals), so the marginals are semiparametric and only the *dependence* is
parametric. Families: ``independence``, ``gaussian``, ``student_t`` (fixed df),
``clayton`` (lower-tail dependence).
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


def _fit_copula(u: np.ndarray, family: str, *, t_df: float):
    """Return (family_used, params) where params feeds :func:`_sample_copula`.

    Any fit failure (or a degenerate parameter) falls back to the independence
    copula, so a bad window never crashes the walk-forward.
    """
    if family == "independence":
        return "independence", {}
    try:
        if family == "gaussian":
            rho = float(np.clip(GaussianCopula(k_dim=2).fit_corr_param(u), -0.999, 0.999))
            return "gaussian", {"rho": rho}
        if family == "student_t":
            rho = float(np.clip(StudentTCopula(k_dim=2).fit_corr_param(u), -0.999, 0.999))
            return "student_t", {"rho": rho, "df": float(t_df)}
        if family == "clayton":
            theta = float(ClaytonCopula().fit_corr_param(u))
            if np.isfinite(theta) and theta > 1e-4:
                return "clayton", {"theta": float(np.clip(theta, 1e-4, 50.0))}
            return "independence", {}          # no lower-tail dependence to model
    except Exception:  # noqa: BLE001 - statsmodels optimiser failure -> independence
        return "independence", {}
    raise ValueError(f"unknown copula family {family!r}")


def _sample_copula(family: str, params: dict, n: int, rng) -> np.ndarray:
    if family == "independence":
        return IndependenceCopula(k_dim=2).rvs(n, rng=rng)
    if family == "gaussian":
        c = np.array([[1.0, params["rho"]], [params["rho"], 1.0]])
        return GaussianCopula(corr=c, k_dim=2).rvs(n, rng=rng)
    if family == "student_t":
        c = np.array([[1.0, params["rho"]], [params["rho"], 1.0]])
        return StudentTCopula(corr=c, df=params["df"], k_dim=2).rvs(n, rng=rng)
    if family == "clayton":
        return ClaytonCopula(theta=params["theta"]).rvs(n, rng=rng)
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
        L = min(len(x) for x in z)
        u = np.column_stack([_pseudo_obs(x[-L:]) for x in z])

        fam, params = _fit_copula(u, self.family, t_df=self.t_df)
        rng = np.random.default_rng(self.seed)
        u_sim = np.clip(_sample_copula(fam, params, self.n_sim, rng), 1e-9, 1 - 1e-9)

        # FHS inverse: empirical quantile of each asset's standardized residuals
        shocks = np.column_stack(
            [np.quantile(z[i][-L:], u_sim[:, i]) for i in range(len(self.assets))]
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
