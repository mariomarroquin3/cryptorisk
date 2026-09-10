"""Merton jump-diffusion (Merton, 1976), V2_PLAN §3.

r = (mu - 0.5 sigma^2 - lam k_bar) + sigma * eps + sum of N ~ Poisson(lam) jumps,
each ~ N(mu_j, sig_j^2). Parameters by MLE on the window (vectorized mixture
likelihood, ported from v1). The one-step predictive distribution is an
:class:`EmpiricalDist` over a large Monte-Carlo sample of next-day returns, so
VaR/ES/CDF work at any level; ``sigma2`` is the closed-form total variance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, poisson

from cryptorisk.models.base import Context, EmpiricalDist, PredictiveDist

_N_MAX = 10
_N_SIM = 20_000


@dataclass(frozen=True)
class JumpParams:
    lam: float = 0.03
    mu_j: float = -0.06
    sig_j: float = 0.10

    def k_bar(self) -> float:
        return float(np.exp(self.mu_j + 0.5 * self.sig_j**2) - 1.0)


def estimate_jump_params(returns: np.ndarray, init: JumpParams | None = None) -> JumpParams:
    r = np.asarray(returns, float)
    mu_base, sig_base = float(r.mean()), float(r.std())
    n_grid = np.arange(_N_MAX + 1)
    init = init or JumpParams()

    def neg_ll(theta):
        lam, mu_j, sig_j = theta
        if lam <= 0 or sig_j <= 0:
            return 1e10
        p_n = poisson.pmf(n_grid, lam)
        k_bar = np.exp(mu_j + 0.5 * sig_j**2) - 1.0
        mu_n = mu_base - lam * k_bar + n_grid * mu_j
        sig_n = np.sqrt(sig_base**2 + n_grid * sig_j**2)
        dens = norm.pdf(r[:, None], mu_n[None, :], sig_n[None, :]) @ p_n
        return -float(np.sum(np.log(np.maximum(dens, 1e-300))))

    try:
        res = minimize(
            neg_ll, [init.lam, init.mu_j, init.sig_j], method="L-BFGS-B",
            bounds=[(1e-4, 0.5), (-0.5, 0.1), (0.01, 0.5)], options={"maxiter": 200},
        )
        if res.success:
            return JumpParams(*res.x)
    except Exception:  # noqa: BLE001
        pass
    return init


class JumpDiffusion:
    name = "Jump-Diffusion"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit_predict(self, ctx: Context) -> PredictiveDist:
        r = ctx.returns
        jp = estimate_jump_params(r)
        rng = np.random.default_rng(self.seed)
        mu, sigma = float(r.mean()), float(r.std())
        k_bar = jp.k_bar()

        eps = rng.standard_normal(_N_SIM)
        n_jumps = rng.poisson(jp.lam, size=_N_SIM)
        jump = np.where(
            n_jumps > 0,
            rng.normal(jp.mu_j * n_jumps, jp.sig_j * np.sqrt(np.maximum(n_jumps, 1))),
            0.0,
        )
        sim = (mu - 0.5 * sigma**2 - jp.lam * k_bar) + sigma * eps + jump
        total_var = sigma**2 + jp.lam * (jp.mu_j**2 + jp.sig_j**2)
        return EmpiricalDist(sample=sim, sigma2_value=float(total_var))
