"""Automated data-quality checks (V2_PLAN §2).

The pipeline runs :func:`check_all` after ingestion and refuses to proceed while
there are unexplained flags (an allow-list of known events lives in
``config/quality_allowlist.yaml`` once we have one).

Checks:

* ``calendar_gap``   - missing calendar day in a daily series
* ``extreme_return`` - |log-return| > 0.40 in a day
* ``zero_volume``    - a day with reported volume == 0
* ``source_divergence`` - |ln(p_a / p_b)| > threshold between two price sources
* ``stale_price``    - identical close on N consecutive days
* ``few_intraday_bars`` - < 80% of the expected 5-min bars on a day
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

_EXPECTED_5M_BARS = 288


@dataclass(frozen=True)
class QualityFlag:
    kind: str
    asset: str
    date: str
    detail: str

    def as_dict(self) -> dict:
        return asdict(self)


def _flags_to_frame(flags: list[QualityFlag]) -> pd.DataFrame:
    if not flags:
        return pd.DataFrame(columns=["kind", "asset", "date", "detail"])
    return pd.DataFrame([f.as_dict() for f in flags]).sort_values(["date", "kind"]).reset_index(drop=True)


def check_daily_series(asset: str, df: pd.DataFrame, *, extreme: float = 0.40,
                       stale_days: int = 4) -> list[QualityFlag]:
    """``df`` columns: date, close, log_return, volume (volume optional)."""
    d = df.sort_values("date").reset_index(drop=True)
    out: list[QualityFlag] = []
    dates = pd.to_datetime(d["date"])

    full = pd.date_range(dates.iloc[0], dates.iloc[-1], freq="D")
    missing = full.difference(dates)
    for m in missing:
        out.append(QualityFlag("calendar_gap", asset, str(m.date()), "no row for this calendar day"))

    lr = d["log_return"].to_numpy()
    for i, v in enumerate(lr):
        if np.isfinite(v) and abs(v) > extreme:
            out.append(QualityFlag("extreme_return", asset, str(pd.Timestamp(d["date"].iloc[i]).date()),
                                   f"log_return={v:+.4f}"))

    if "volume" in d.columns:
        for i, v in enumerate(d["volume"].to_numpy()):
            if v is not None and np.isfinite(v) and v == 0:
                out.append(QualityFlag("zero_volume", asset, str(pd.Timestamp(d["date"].iloc[i]).date()), "volume == 0"))

    close = d["close"].to_numpy()
    run = 1
    for i in range(1, len(close)):
        run = run + 1 if close[i] == close[i - 1] else 1
        if run == stale_days:
            out.append(QualityFlag("stale_price", asset, str(pd.Timestamp(d["date"].iloc[i]).date()),
                                   f"{stale_days} identical closes"))
    return out


def check_source_divergence(asset: str, a: pd.DataFrame, b: pd.DataFrame, *,
                            threshold: float = 0.02) -> list[QualityFlag]:
    """``a``, ``b`` columns: date, close. Flags days where the two sources
    disagree by more than ``threshold`` in log terms."""
    m = a[["date", "close"]].merge(b[["date", "close"]], on="date", suffixes=("_a", "_b"))
    div = np.log(m["close_a"] / m["close_b"]).abs()
    out = []
    for i, v in enumerate(div.to_numpy()):
        if np.isfinite(v) and v > threshold:
            out.append(QualityFlag("source_divergence", asset, str(pd.Timestamp(m["date"].iloc[i]).date()),
                                   f"|ln(a/b)|={v:.4f} > {threshold}"))
    return out


def check_intraday_coverage(asset: str, realized: pd.DataFrame, *, min_frac: float = 0.80) -> list[QualityFlag]:
    """``realized`` columns include date, n_bars."""
    out = []
    for _, row in realized.iterrows():
        frac = row["n_bars"] / _EXPECTED_5M_BARS
        if frac < min_frac:
            out.append(QualityFlag("few_intraday_bars", asset, str(pd.Timestamp(row["date"]).date()),
                                   f"{int(row['n_bars'])}/{_EXPECTED_5M_BARS} bars ({frac:.0%})"))
    return out


def check_all(
    per_asset_daily: dict[str, pd.DataFrame],
    per_asset_sources: dict[str, tuple[pd.DataFrame, pd.DataFrame]] | None = None,
    per_asset_realized: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    flags: list[QualityFlag] = []
    for asset, df in per_asset_daily.items():
        flags += check_daily_series(asset, df)
    if per_asset_sources:
        for asset, (a, b) in per_asset_sources.items():
            flags += check_source_divergence(asset, a, b)
    if per_asset_realized:
        for asset, rdf in per_asset_realized.items():
            flags += check_intraday_coverage(asset, rdf)
    return _flags_to_frame(flags)
