"""Realized measures from 5-minute bars (V2_PLAN §2, §3).

From a continuous 24/7 series of 5-minute closes (crypto has no overnight gap):

* ``rv``       realized variance,  Sum r_i^2
* ``bv``       bipower variation,  (pi/2) * (M/(M-1)) * Sum |r_i||r_{i-1}|
* ``rsv_pos``  / ``rsv_neg``  realized semivariance (rsv_pos + rsv_neg = rv)
* ``jump``     jump component: ``max(rv - bv, 0)`` gated by the
  Barndorff-Nielsen & Shephard (2006) Z-test at 5% (0 if not significant),
  or by Lee & Mykland (2008) when ``jump_test="LM"`` (per-bar detection,
  summed to daily)
* ``n_bars``   intraday returns used that day

Cleaning follows Barndorff-Nielsen, Hansen, Lunde & Shephard (2009), adapted:
drop duplicate timestamps (keep last), non-positive prices, and 5-minute
returns that are MAD-outliers (|r| > k * 1.4826 * MAD of a rolling window) or
implausibly large (> 50% in 5 minutes).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

_MU1 = np.sqrt(2.0 / np.pi)                       # E|Z|
_BV_C = np.pi / 2.0                               # _MU1 ** -2
_MU_43 = 2.0 ** (2 / 3) * 0.9027452929509336      # 2^{2/3} Gamma(7/6)/Gamma(1/2)
_TQ_C = _MU_43 ** -3
_THETA = np.pi**2 / 4.0 + np.pi - 5.0             # ~= 0.6090


def clean_bars(bars: pd.DataFrame, *, mad_window: int = 120, mad_k: float = 10.0) -> pd.DataFrame:
    """Return ``bars`` sorted by ``ts`` with bad rows removed. Expects columns
    ``ts`` (datetime64) and ``close`` (>0)."""
    b = bars.loc[:, ["ts", "close"]].dropna()
    b = b.sort_values("ts").drop_duplicates("ts", keep="last")
    b = b[b["close"] > 0].reset_index(drop=True)
    if len(b) < 3:
        return b

    r = np.log(b["close"].to_numpy())
    dr = np.diff(r, prepend=r[0])
    s = pd.Series(dr)
    med = s.rolling(mad_window, min_periods=20, center=True).median()
    mad = (s - med).abs().rolling(mad_window, min_periods=20, center=True).median()
    thr = mad_k * 1.4826 * mad.bfill().ffill()
    keep = ((s - med).abs() <= thr.clip(lower=1e-6)) & (s.abs() <= 0.5)
    keep.iloc[0] = True
    return b.loc[keep.to_numpy()].reset_index(drop=True)


def _daily_returns(bars: pd.DataFrame) -> pd.DataFrame:
    """One row per 5-min return, tagged with the DATE of its END timestamp."""
    b = bars.sort_values("ts")
    r = np.diff(np.log(b["close"].to_numpy()))
    ts_end = b["ts"].to_numpy()[1:]
    return pd.DataFrame({"date": pd.to_datetime(ts_end).normalize(), "r": r})


def _bns_zstat(r: np.ndarray) -> float:
    """BNS (2006) jump Z-statistic for one day of intraday returns."""
    m = r.size
    if m < 4:
        return 0.0
    rv = np.sum(r**2)
    bv = _BV_C * (m / (m - 1)) * np.sum(np.abs(r[1:]) * np.abs(r[:-1]))
    if bv <= 0 or rv <= 0:
        return 0.0
    a = np.abs(r) ** (4 / 3)
    tq = m * _TQ_C * np.sum(a[2:] * a[1:-1] * a[:-2])
    denom = _THETA * max(1.0, tq / bv**2) / m
    return float((rv - bv) / rv / np.sqrt(denom)) if denom > 0 else 0.0


def _lm_jump_mask(r: np.ndarray, *, alpha: float = 0.01) -> np.ndarray:
    """Lee & Mykland (2008) per-bar jump indicator."""
    m = r.size
    if m < 30:
        return np.zeros(m, dtype=bool)
    k = max(int(np.sqrt(252 * m)), 20)
    absr = pd.Series(np.abs(r))
    sigma = (
        (_BV_C * absr * absr.shift(1))
        .rolling(k, min_periods=k // 2)
        .mean()
        .pow(0.5)
        .bfill()
        .to_numpy()
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = np.abs(r) / np.where(sigma > 0, sigma, np.nan)
    c = np.sqrt(2 / np.pi)
    sn = 1.0 / (c * np.sqrt(2 * np.log(m)))
    cn = np.sqrt(2 * np.log(m)) / c - (np.log(np.pi) + np.log(np.log(m))) / (2 * c * np.sqrt(2 * np.log(m)))
    crit = -np.log(-np.log(1 - alpha)) * sn + cn
    return np.nan_to_num(stat, nan=0.0) > crit


def realized_daily(
    bars_5m: pd.DataFrame,
    *,
    subsample: bool = True,
    subsample_step: int = 2,
    jump_test: str = "BNS",
) -> pd.DataFrame:
    """Daily realized measures. ``bars_5m`` needs columns ``ts`` and ``close``
    (raw; cleaned internally). When ``subsample`` is set, ``rv`` is the average
    of RV over ``subsample_step`` staggered coarser grids (noise reduction)."""
    b = clean_bars(bars_5m)
    if b.empty:
        return _empty()

    rets = _daily_returns(b)
    rows = []
    for day, g in rets.groupby("date", sort=True):
        r = g["r"].to_numpy()
        m = r.size
        if m < 2:
            continue

        if subsample and subsample_step > 1:
            q = subsample_step
            rv_grids = []
            for o in range(q):
                seg = r[o:]
                n = (seg.size // q) * q
                if n == 0:
                    continue
                agg = seg[:n].reshape(-1, q).sum(axis=1)
                rv_grids.append(float(np.sum(agg**2)))
            rv = float(np.mean(rv_grids)) if rv_grids else float(np.sum(r**2))
        else:
            rv = float(np.sum(r**2))

        bv = float(_BV_C * (m / (m - 1)) * np.sum(np.abs(r[1:]) * np.abs(r[:-1]))) if m > 1 else np.nan
        rsv_pos = float(np.sum(r[r > 0] ** 2))
        rsv_neg = float(np.sum(r[r < 0] ** 2))

        if jump_test.upper() == "LM":
            jm = _lm_jump_mask(r)
            jump = float(np.sum(r[jm] ** 2))
        else:  # BNS
            z = _bns_zstat(r)
            jump = float(max(np.sum(r**2) - bv, 0.0)) if z > norm.ppf(0.95) else 0.0

        rows.append((pd.Timestamp(day).date(), rv, bv, rsv_pos, rsv_neg, jump, int(m)))

    if not rows:
        return _empty()
    return pd.DataFrame(rows, columns=["date", "rv", "bv", "rsv_pos", "rsv_neg", "jump", "n_bars"])


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "rv", "bv", "rsv_pos", "rsv_neg", "jump", "n_bars"])
