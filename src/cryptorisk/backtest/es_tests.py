"""Expected Shortfall backtests (V2_PLAN §5.2).

* Acerbi-Szekely (2014) Z1 and Z2 tests; p-values by simulation under the
  model's predictive distribution.
* ES-regression backtest (Bayer & Dimitriadis, 2022).

Phase 3.
"""

from __future__ import annotations

import numpy as np


def acerbi_szekely_z2(realized: np.ndarray, var: np.ndarray, es: np.ndarray, alpha: float) -> float:
    """Z2 statistic. E[Z2] = 0 under a correctly specified ES."""
    raise NotImplementedError("es_tests.acerbi_szekely_z2 - Phase 3")


def acerbi_szekely_z1(realized: np.ndarray, var: np.ndarray, es: np.ndarray) -> float:
    raise NotImplementedError("es_tests.acerbi_szekely_z1 - Phase 3")
