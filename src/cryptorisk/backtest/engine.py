"""Walk-forward backtesting engine (V2_PLAN §4).

Model-agnostic. Given a return series (+ optional realized / exog blocks), a
model, a window scheme and a refit cadence, produces one
:class:`~cryptorisk.models.base.PredictiveDist` per out-of-sample day and
records VaR/ES/sigma2/realized for each configured alpha.

To be ported from v1 ``risk_models/walk_forward.py`` (hardened + parallelised)
in Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cryptorisk.models.base import Model


@dataclass
class WalkForwardResult:
    """Tidy long frame: one row per (date, alpha)."""

    frame: pd.DataFrame  # columns: date, asset, model, alpha, var, es, sigma2, realized, violation


def walk_forward(
    returns: pd.Series,
    dates: pd.Series,
    model: Model,
    *,
    alphas: list[float],
    window: int | str = 500,
    oos_start: np.datetime64 | str | None = None,
    refit_every: int = 1,
    realized: dict[str, pd.Series] | None = None,
    exog: dict[str, pd.Series] | None = None,
    n_jobs: int = 1,
) -> WalkForwardResult:
    raise NotImplementedError("engine.walk_forward - Phase 2 (port from v1 walk_forward.py)")
