"""Historical Simulation and age-weighted HS (V2_PLAN §3).

* ``HistoricalSimulation()`` - empirical quantile of the window.
* ``HistoricalSimulation(halflife=N)`` - Boudoukh, Richardson & Whitelaw (1998):
  exponentially decaying weights, most recent observation heaviest.
"""

from __future__ import annotations

import numpy as np

from cryptorisk.models.base import Context, EmpiricalDist, PredictiveDist


class HistoricalSimulation:
    def __init__(self, halflife: int | None = None, name: str | None = None):
        self.halflife = halflife
        self.name = name or ("AWHS" if halflife else "HS")

    def fit_predict(self, ctx: Context) -> PredictiveDist:
        r = ctx.returns
        if self.halflife is None:
            return EmpiricalDist(r)
        age = np.arange(r.size)[::-1]  # oldest -> largest age
        w = 0.5 ** (age / float(self.halflife))
        return EmpiricalDist(r, weights=w)
