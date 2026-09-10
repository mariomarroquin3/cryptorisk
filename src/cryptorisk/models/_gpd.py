"""Peaks-Over-Threshold GPD tail distribution.

Given a sample of standardized innovations ``z`` and a one-step location/scale
(``loc``, ``scale``), fits a Generalized Pareto Distribution to the lower tail of
``z`` (Peaks-Over-Threshold at ``threshold_q``) and exposes a full
:class:`~cryptorisk.models.base.PredictiveDist`:

* tail: the GPD-POT quantile / ES formulae (McNeil & Frey, 2000)
* body: the empirical CDF of ``z``

If the fit degenerates or ``alpha >= n_u/n`` (the quantile is inside the body),
:meth:`var` / :meth:`es` fall back to the empirical quantile of ``z``.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import genpareto

from cryptorisk.models.base import PredictiveDist


class GpdTailDist(PredictiveDist):
    def __init__(self, z: np.ndarray, *, loc: float = 0.0, scale: float = 1.0,
                 threshold_q: float = 0.90):
        z = np.asarray(z, float)
        self.z = np.sort(z[np.isfinite(z)])
        self.loc = float(loc)
        self.scale = float(scale)
        losses = -self.z
        self.u = float(np.quantile(losses, threshold_q))
        exc = losses[losses > self.u] - self.u
        self.p_exceed = exc.size / self.z.size if self.z.size else 0.0
        self.ok = exc.size >= 10 and self.p_exceed > 0
        if self.ok:
            try:
                self.xi, _, self.beta = genpareto.fit(exc, floc=0)
                self.ok = np.isfinite(self.xi) and self.beta > 0
            except Exception:  # noqa: BLE001
                self.ok = False
        if not self.ok:
            self.xi = self.beta = np.nan

    # -- helpers ----------------------------------------------------------- #
    def _emp_q(self, alpha: float) -> float:
        return float(np.quantile(self.z, alpha))

    def _var_loss(self, alpha: float) -> float | None:
        if not self.ok:
            return None
        tail_prob = alpha / self.p_exceed
        if tail_prob >= 1.0:
            return None
        if abs(self.xi) < 1e-8:
            vl = self.u + self.beta * np.log(1.0 / tail_prob)
        else:
            vl = self.u + (self.beta / self.xi) * (tail_prob ** (-self.xi) - 1.0)
        return vl if np.isfinite(vl) and vl > 0 else None

    # -- PredictiveDist -------------------------------------------------- #
    def var(self, alpha: float) -> float:
        self._check_alpha(alpha)
        vl = self._var_loss(alpha)
        z_q = -vl if vl is not None else self._emp_q(alpha)
        return self.loc + self.scale * z_q

    def es(self, alpha: float) -> float:
        self._check_alpha(alpha)
        vl = self._var_loss(alpha)
        if vl is None:
            q = self._emp_q(alpha)
            tail = self.z[self.z <= q]
            z_e = float(tail.mean()) if tail.size else q
        elif self.xi < 1:
            el = (vl + self.beta - self.xi * self.u) / (1.0 - self.xi)
            z_e = -max(el, vl)
        else:
            z_e = -2.0 * vl
        return self.loc + self.scale * z_e

    def sigma2(self) -> float:
        return self.scale**2

    def cdf(self, x: float) -> float:
        z = (x - self.loc) / self.scale
        loss = -z
        if not self.ok or loss <= self.u:
            return float(np.searchsorted(self.z, z, side="right") / self.z.size)
        base = 1.0 + self.xi * (loss - self.u) / self.beta
        if base <= 0.0:  # beyond the GPD's finite upper endpoint (xi < 0)
            return 0.0
        return float(np.clip(self.p_exceed * base ** (-1.0 / self.xi), 0.0, 1.0))

    def ppf(self, u: float) -> float:
        if not 0.0 < u < 1.0:
            raise ValueError("u must be in (0, 1)")
        return self.var(u) if u < self.p_exceed else self.loc + self.scale * self._emp_q(u)
