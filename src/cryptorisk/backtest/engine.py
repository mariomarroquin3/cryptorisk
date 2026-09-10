"""Walk-forward backtesting engine (V2_PLAN §4).

Model-agnostic. Given a per-asset frame (``date``, ``log_return``, optional
realized / exog columns), a :class:`~cryptorisk.models.base.Model`, a window
scheme and the OOS start, it produces one
:class:`~cryptorisk.models.base.PredictiveDist` per out-of-sample day and
records, for each configured ``alpha``: VaR, ES, sigma2, the realized return,
the violation flag, and the PIT value.

Parameters that shape a run live in ``config/study.yaml`` (§4). The engine is
sequential; the study layer parallelises across (model, asset, window).

``refit_every`` reuses the previous day's :class:`PredictiveDist` for the
in-between days - only relevant for expensive models (MS-GARCH). The Phase-2a
models are cheap and run with ``refit_every=1``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cryptorisk.models.base import Context, Model, PredictiveDist

_REALIZED_COLS = ("rv", "bv", "rsv_pos", "rsv_neg", "jump", "rq")


@dataclass
class WalkForwardResult:
    """Tidy long frame, one row per (date, alpha)."""

    frame: pd.DataFrame

    def hit_rate(self, alpha: float) -> float:
        s = self.frame.loc[self.frame["alpha"] == alpha, "violation"]
        return float(s.mean())

    def n_obs(self, alpha: float) -> int:
        return int((self.frame["alpha"] == alpha).sum())


def _safe(fn, *a):
    try:
        v = fn(*a)
        return float(v) if v is not None and np.isfinite(v) else np.nan
    except (NotImplementedError, ValueError, FloatingPointError, ZeroDivisionError):
        return np.nan


def walk_forward(
    df: pd.DataFrame,
    model: Model,
    *,
    alphas: list[float],
    asset: str = "BTC",
    window: int | str = 500,
    oos_start: str | pd.Timestamp | None = None,
    refit_every: int = 1,
    exog_cols: tuple[str, ...] = (),
) -> WalkForwardResult:
    d = df.sort_values("date").reset_index(drop=True)
    d["date"] = pd.to_datetime(d["date"])
    d = d[np.isfinite(d["log_return"])].reset_index(drop=True)

    r = d["log_return"].to_numpy(float)
    dates = d["date"].to_numpy()
    n = len(d)

    if isinstance(window, str) and window != "expanding":
        raise ValueError("window must be an int or 'expanding'")
    min_train = 250 if window == "expanding" else int(window)
    if n <= min_train + 1:
        raise ValueError(f"not enough data: {n} rows for min train {min_train}")

    start_idx = min_train
    if oos_start is not None:
        os_ts = np.datetime64(pd.Timestamp(oos_start))
        start_idx = max(start_idx, int(np.searchsorted(dates, os_ts)))

    realized_present = [c for c in _REALIZED_COLS if c in d.columns]
    rows: list[dict] = []
    last: PredictiveDist | None = None

    for t in range(start_idx, n):
        lo = 0 if window == "expanding" else t - int(window)
        if (t - start_idx) % refit_every == 0 or last is None:
            realized = (
                {c: d[c].to_numpy(float)[lo:t] for c in realized_present} if realized_present else None
            )
            exog = {c: d[c].to_numpy(float)[lo:t] for c in exog_cols} if exog_cols else None
            ctx = Context(
                returns=r[lo:t], dates=dates[lo:t], asof=dates[t - 1],
                asset=asset, realized=realized, exog=exog,
            )
            last = model.fit_predict(ctx)

        realized_ret = float(r[t])
        for a in alphas:
            var = _safe(last.var, a)
            es = _safe(last.es, a)
            rows.append({
                "date": pd.Timestamp(dates[t]),
                "asset": asset,
                "model": model.name,
                "window": window,
                "alpha": a,
                "var": var,
                "es": es,
                "sigma2": _safe(last.sigma2),
                "realized": realized_ret,
                "violation": bool(realized_ret < var) if np.isfinite(var) else np.nan,
                "pit": _safe(last.cdf, realized_ret),
            })

    return WalkForwardResult(pd.DataFrame(rows))
