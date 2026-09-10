"""Comparative scoring: consistent losses + Model Confidence Set (V2_PLAN §5.3).

* ``fz0_loss`` - Fissler-Ziegel 0-homogeneous joint (VaR, ES) loss, strictly
  consistent at level ``alpha`` (Patton, Ziegel & Chen, 2019). Lower is better.
* ``qlike_loss`` - QLIKE loss for variance forecasts vs a realized proxy
  (V2_PLAN §5.4).
* ``diebold_mariano`` - paired equal-predictive-ability test on a loss diff.
* ``model_confidence_set`` - Hansen, Lunde & Nason (2011); returns the set of
  models not significantly worse than the best, at ``confidence``.

Bodies land in Phase 3. Signatures are fixed here so downstream code can be
written against them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def fz0_loss(realized: np.ndarray, var: np.ndarray, es: np.ndarray, alpha: float) -> np.ndarray:
    r"""Per-observation FZ0 loss for the (VaR, ES) pair at level ``alpha``.

    With :math:`v_t = \mathrm{VaR}_t < 0`, :math:`e_t = \mathrm{ES}_t < 0`:

    .. math::
        L^{FZ0}_t = \frac{1}{\alpha e_t}\,\mathbf{1}\{r_t \le v_t\}\,(v_t - r_t)
                    + \frac{v_t}{e_t} + \ln(-e_t) - 1
    """
    raise NotImplementedError("scoring.fz0_loss - Phase 3")


def qlike_loss(sigma2_forecast: np.ndarray, realized_var: np.ndarray) -> np.ndarray:
    """QLIKE = rv/h - ln(rv/h) - 1 ; robust to a noisy variance proxy."""
    raise NotImplementedError("scoring.qlike_loss - Phase 3")


@dataclass(frozen=True)
class DMResult:
    statistic: float
    p_value: float


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, *, hac_lag: int | None = None) -> DMResult:
    raise NotImplementedError("scoring.diebold_mariano - Phase 3")


@dataclass(frozen=True)
class MCSResult:
    included: list[str]          # models in the confidence set
    p_values: dict[str, float]   # MCS p-value per model
    best: str


def model_confidence_set(
    losses: dict[str, np.ndarray],
    *,
    confidence: float = 0.90,
    block_len: int = 20,
    n_boot: int = 5000,
    seed: int | None = None,
) -> MCSResult:
    raise NotImplementedError("scoring.model_confidence_set - Phase 3")
