"""EWMA / RiskMetrics volatility (V2_PLAN §3).

sigma2_{t+1|t} = lam * sigma2_{t|t-1} + (1 - lam) * r_t^2 , lam = 0.94 (J.P.
Morgan, 1996). Zero conditional mean. Normal innovations by default; ``dist="t"``
uses a fixed-nu standardized Student-t for fatter tails.
"""

from __future__ import annotations

import numpy as np

from cryptorisk.models._dist import normal_z, student_t_z
from cryptorisk.models.base import Context, ParametricDist, PredictiveDist


class EWMA:
    def __init__(self, lam: float = 0.94, dist: str = "normal", nu: float = 6.0, name: str | None = None):
        if not 0.5 < lam < 1.0:
            raise ValueError("lam should be in (0.5, 1.0)")
        self.lam = lam
        self.dist = dist
        self.nu = nu
        self.name = name or ("EWMA-t" if dist == "t" else "EWMA")

    def fit_predict(self, ctx: Context) -> PredictiveDist:
        r = ctx.returns
        s2 = float(np.mean(r[: min(30, r.size)] ** 2))  # seed on the first month
        for x in r:
            s2 = self.lam * s2 + (1.0 - self.lam) * float(x) ** 2
        sigma = float(np.sqrt(max(s2, 1e-16)))
        ppf, cdf, es = student_t_z(self.nu) if self.dist == "t" else normal_z()
        return ParametricDist(loc=0.0, scale=sigma, z_ppf=ppf, z_cdf=cdf, z_es=es)
