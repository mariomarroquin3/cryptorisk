"""Quantile Regression Forest (Meinshausen, 2006) for one-step VaR/ES.

A plain ``RandomForestRegressor`` already keeps, per leaf, exactly which
training observations landed there. Meinshausen's insight: instead of
collapsing a leaf to its mean (ordinary regression), use the leaf's full set
of training targets as a local empirical distribution, weight it by how often
the test point shares a leaf with each training row across all trees, and
read off *any* quantile of that weighted distribution -- not just the mean.
That gives the whole one-step conditional distribution of the return, so
VaR/ES fall out directly as weighted quantiles/tail-means (:class:`EmpiricalDist`)
with no assumption about the tail's shape (unlike the Student-t tail every
GARCH-family model in this suite relies on).

Features are HAR-style (Corsi, 2009) but built from squared returns alone, so
the model runs on every asset/window (no realized-measure dependency like
HAR-RV) and adds the realized measure as extra features when available:
lag-1 return and squared return, a leverage term (squared *down* return
only), and 5-/22-day rolling means of squared returns (and of log RV, when
present) as short/medium volatility-regime signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from cryptorisk.models._util import ffill
from cryptorisk.models.base import Context, EmpiricalDist, PredictiveDist

_W, _M = 5, 22
_MIN_TRAIN = 120
# Matches config/study.yaml's global `seed` -- the forest's bootstrap/feature
# subsampling is the only randomness in this model, so it is pinned for
# reproducibility rather than reading the config (called once per day; no
# need to couple this model to config I/O for a single constant).
_SEED = 20260101


def _feature_frame(r: np.ndarray, rlz) -> pd.DataFrame:
    """One row per day, length ``n + 1``: rows 0..n-1 line up with targets
    ``r[0..n-1]``, row ``n`` is the one-step-ahead forecast row. Every column
    is a ``.shift(1)`` of same-day-or-earlier information, so row ``t``'s
    features only use data through ``t - 1`` -- including the forecast row,
    which uses ``r[n-1]`` etc. exactly like a real one-step forecast would.
    """
    r_ext = pd.Series(np.append(r, np.nan))
    r2 = r_ext**2
    feats = {
        "r_lag1": r_ext.shift(1),
        "r2_lag1": r2.shift(1),
        "r2_w": r2.shift(1).rolling(_W).mean(),
        "r2_m": r2.shift(1).rolling(_M).mean(),
        "down2_lag1": r_ext.shift(1).clip(upper=0.0) ** 2,
    }
    if rlz is not None and "rv" in rlz:
        rv = np.asarray(rlz["rv"], float)
        rv = np.where(np.isfinite(rv) & (rv > 0), rv, np.nan)
        if np.isnan(rv).mean() <= 0.2:
            log_rv = pd.Series(np.log(np.append(ffill(rv), np.nan)))
            feats["log_rv_lag1"] = log_rv.shift(1)
            feats["log_rv_w"] = log_rv.shift(1).rolling(_W).mean()
            feats["log_rv_m"] = log_rv.shift(1).rolling(_M).mean()
    return pd.DataFrame(feats)


def _qrf_weights(rf: RandomForestRegressor, x_train: np.ndarray, x_fcast: np.ndarray) -> np.ndarray:
    """Per-training-row weight (Meinshausen, 2006 eq. 4): for each tree, split
    weight 1/(#trees) evenly among the training rows sharing the forecast
    point's leaf, then average over trees. Vectorised over trees -- this runs
    once per day, so an O(n_train x n_trees) boolean comparison must stay
    cheap rather than looping in Python per tree.
    """
    leaves_train = rf.apply(x_train)  # (n_train, n_trees)
    leaf_fcast = rf.apply(x_fcast.reshape(1, -1))[0]  # (n_trees,)
    matches = leaves_train == leaf_fcast[None, :]
    counts = matches.sum(axis=0)  # > 0 always: the forecast leaf came from this same tree
    return (matches / counts[None, :]).sum(axis=1) / rf.n_estimators


class RandomForestQR:
    """Quantile Regression Forest: VaR/ES from the forest's own weighted
    empirical distribution of next-day returns, not a parametric tail."""

    def __init__(
        self,
        n_estimators: int = 200,
        min_samples_leaf: int = 20,
        name: str | None = None,
    ):
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.name = name or "RF-QR"

    def fit_predict(self, ctx: Context) -> PredictiveDist:
        r = ctx.returns
        n = r.size
        feats = _feature_frame(r, ctx.realized)
        train_mask = feats.iloc[:n].notna().all(axis=1).to_numpy()
        x_fcast = feats.iloc[n].to_numpy()
        if train_mask.sum() < _MIN_TRAIN or not np.all(np.isfinite(x_fcast)):
            return EmpiricalDist(r)

        x_train = feats.iloc[:n].to_numpy()[train_mask]
        y_train = r[train_mask]

        try:
            rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                min_samples_leaf=self.min_samples_leaf,
                random_state=_SEED,
                n_jobs=1,
            )
            rf.fit(x_train, y_train)
            weights = _qrf_weights(rf, x_train, x_fcast)
        except (ValueError, FloatingPointError):
            return EmpiricalDist(r)

        return EmpiricalDist(sample=y_train, weights=weights, sigma2_value=float(np.var(y_train)))
