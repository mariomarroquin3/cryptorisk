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

    def _fit(self, ctx: Context):
        """Fit the forest on the window. Returns ``(rf, feature_names, x_train,
        y_train, x_fcast, weights)`` or ``None`` when the window can't support a
        fit (short window / non-finite forecast row / sklearn failure)."""
        r = ctx.returns
        n = r.size
        feats = _feature_frame(r, ctx.realized)
        train_mask = feats.iloc[:n].notna().all(axis=1).to_numpy()
        x_fcast = feats.iloc[n].to_numpy()
        if train_mask.sum() < _MIN_TRAIN or not np.all(np.isfinite(x_fcast)):
            return None
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
            return None
        return rf, list(feats.columns), x_train, y_train, x_fcast, weights

    def fit_predict(self, ctx: Context) -> PredictiveDist:
        fit = self._fit(ctx)
        if fit is None:
            return EmpiricalDist(ctx.returns)
        _, _, _, y_train, _, weights = fit
        # Conditional variance under the forest's own weights -- NOT the plain
        # window variance, which is the same for every day's forecast and would
        # make the volatility-forecast evaluation (QLIKE, Mincer-Zarnowitz) score
        # an unconditional number.
        mu = float(np.sum(weights * y_train))
        var = float(np.sum(weights * (y_train - mu) ** 2))
        return EmpiricalDist(sample=y_train, weights=weights, sigma2_value=var)

    def explain(self, ctx: Context, alpha: float = 0.025) -> dict | None:
        """Why this forecast: per-feature impurity importance, the effective
        number of training days behind the forecast (``1/sum(w^2)``), the
        conditional VaR vs the flat-weight (Historical-Simulation) VaR on the
        same window, and the forecast row's features as z-scores of the window.
        """
        fit = self._fit(ctx)
        if fit is None:
            return None
        rf, names, x_train, y_train, x_fcast, weights = fit
        dist = EmpiricalDist(sample=y_train, weights=weights)
        sd = x_train.std(axis=0)
        z = (x_fcast - x_train.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
        return {
            "features": names,
            "importance": rf.feature_importances_.tolist(),
            "ess": float(1.0 / np.sum(weights**2)),
            "n_train": int(y_train.size),
            "var_cond": dist.var(alpha),
            "var_hs": EmpiricalDist(sample=y_train).var(alpha),
            "inputs": x_fcast.tolist(),
            "zscores": z.tolist(),
        }

    def partial_dependence(
        self, ctx: Context, feature: str, alpha: float = 0.025, n_points: int = 15
    ) -> dict | None:
        """How today's VaR would change if just ``feature`` were different,
        every other input held at its actual value from this forecast row (an
        individual conditional expectation curve, not an average over the
        training sample). Sweeps ``feature`` across its 1st-99th percentile
        range in the training window, refits nothing -- ``_qrf_weights`` only
        needs the forest's existing leaf structure, so each grid point is a
        cheap re-weighting, not a new fit. ``None`` when the window is too
        short or ``feature`` isn't one of this window's inputs (the realized
        -measure features are absent without RV)."""
        fit = self._fit(ctx)
        if fit is None:
            return None
        rf, names, x_train, y_train, x_fcast, _ = fit
        if feature not in names:
            return None
        idx = names.index(feature)
        lo, hi = np.percentile(x_train[:, idx], [1, 99])
        if not (hi > lo):
            return None
        grid = np.linspace(lo, hi, n_points)
        var_grid = []
        for g in grid:
            xg = x_fcast.copy()
            xg[idx] = g
            w = _qrf_weights(rf, x_train, xg)
            var_grid.append(EmpiricalDist(sample=y_train, weights=w).var(alpha))
        return {
            "feature": feature,
            "grid": grid.tolist(),
            "var": var_grid,
            "actual": float(x_fcast[idx]),
        }
