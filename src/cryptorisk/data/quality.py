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
* ``stale_realized``  - realized measures end > 3 days before the daily returns
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

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
                            threshold: float = 0.02,
                            returns: pd.DataFrame | None = None) -> list[QualityFlag]:
    """``a``, ``b`` columns: date, close. Flags a day when the two price sources
    disagree by more than ``threshold`` in log terms **and** by more than that
    day's own log-return (``returns`` columns: date, log_return). The second
    condition removes timing artefacts: on a big-move day an exchange close and
    a reference-rate snapshot naturally differ by a few percent. If a source
    disagrees by more than the asset actually moved, that is a real problem."""
    m = a[["date", "close"]].merge(b[["date", "close"]], on="date", suffixes=("_a", "_b"))
    if returns is not None:
        m = m.merge(returns[["date", "log_return"]], on="date", how="left")
    div = np.log(m["close_a"] / m["close_b"]).abs()
    day_move = m["log_return"].abs() if "log_return" in m.columns else pd.Series(0.0, index=m.index)
    out = []
    for i, v in enumerate(div.to_numpy()):
        mv = day_move.iloc[i]
        if np.isfinite(v) and v > threshold and (not np.isfinite(mv) or v > mv):
            out.append(QualityFlag("source_divergence", asset, str(pd.Timestamp(m["date"].iloc[i]).date()),
                                   f"|ln(a/b)|={v:.4f} > max({threshold}, day |ret|={mv:.4f})"))
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


def check_realized_lag(
    asset: str, daily: pd.DataFrame, realized: pd.DataFrame, *, max_lag_days: int = 3
) -> list[QualityFlag]:
    """Realized measures ending well before the daily returns: the models that
    use them then see a forward-filled (stale) RV over the tail of the sample."""
    if daily.empty or realized.empty:
        return []
    last_r, last_rv = pd.Timestamp(daily["date"].max()), pd.Timestamp(realized["date"].max())
    lag = (last_r - last_rv).days
    if lag <= max_lag_days:
        return []
    return [QualityFlag("stale_realized", asset, str(last_rv.date()),
                        f"realized measures end {lag} days before the daily returns "
                        f"({last_rv.date()} vs {last_r.date()}); RV is forward-filled over the gap")]


def check_all(
    per_asset_daily: dict[str, pd.DataFrame],
    per_asset_sources: dict[str, tuple[pd.DataFrame, pd.DataFrame]] | None = None,
    per_asset_realized: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    flags: list[QualityFlag] = []
    for asset, df in per_asset_daily.items():
        flags += check_daily_series(asset, df)
        if per_asset_realized and asset in per_asset_realized:
            flags += check_realized_lag(asset, df, per_asset_realized[asset])
    if per_asset_sources:
        for asset, (a, b) in per_asset_sources.items():
            flags += check_source_divergence(asset, a, b, returns=per_asset_daily.get(asset))
    if per_asset_realized:
        for asset, rdf in per_asset_realized.items():
            flags += check_intraday_coverage(asset, rdf)
    return _flags_to_frame(flags)


def load_allowlist(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return yaml.safe_load(p.read_text(encoding="utf-8")) or []


def apply_allowlist(report: pd.DataFrame, rules: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``report`` into (unexplained, explained). A row is explained if it
    matches a rule on ``kind`` plus any of: ``dates`` (exact), ``before`` /
    ``after`` (date bound), ``asset``. Every rule must carry a ``reason``."""
    if report.empty or not rules:
        return report, report.iloc[0:0].assign(reason=pd.Series(dtype=str))

    explained_idx: set[int] = set()
    reasons: dict[int, str] = {}
    d = pd.to_datetime(report["date"])
    for rule in rules:
        if "reason" not in rule:
            raise ValueError(f"allowlist rule without a reason: {rule}")
        m = report["kind"] == rule["kind"]
        if "asset" in rule:
            m &= report["asset"] == rule["asset"]
        if "dates" in rule:
            m &= report["date"].isin([str(x) for x in rule["dates"]])
        if "before" in rule:
            m &= d < pd.Timestamp(rule["before"])
        if "after" in rule:
            m &= d >= pd.Timestamp(rule["after"])
        for i in report.index[m]:
            explained_idx.add(i)
            reasons.setdefault(i, " ".join(rule["reason"].split()))

    explained = report.loc[sorted(explained_idx)].copy()
    explained["reason"] = [reasons[i] for i in explained.index]
    unexplained = report.drop(index=explained_idx)
    return unexplained.reset_index(drop=True), explained.reset_index(drop=True)
