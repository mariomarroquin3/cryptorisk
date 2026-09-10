"""Data-quality checks: each flag fires on a crafted defect."""

import numpy as np
import pandas as pd

from cryptorisk.data import quality


def _daily(closes, dates=None, volume=None):
    n = len(closes)
    dates = dates if dates is not None else pd.date_range("2021-01-01", periods=n, freq="D")
    lr = np.concatenate([[np.nan], np.diff(np.log(closes))])
    d = pd.DataFrame({"date": dates, "close": closes, "log_return": lr})
    if volume is not None:
        d["volume"] = volume
    return d


def test_calendar_gap_flagged():
    dates = list(pd.date_range("2021-01-01", periods=5, freq="D"))
    dates.pop(2)  # drop 2021-01-03
    df = _daily(np.linspace(100, 104, 4), dates=pd.DatetimeIndex(dates))
    kinds = {f.kind for f in quality.check_daily_series("BTC", df)}
    assert "calendar_gap" in kinds


def test_extreme_return_flagged():
    df = _daily(np.array([100.0, 100.0, 160.0, 160.0]))  # +47% jump
    flags = quality.check_daily_series("BTC", df)
    assert any(f.kind == "extreme_return" for f in flags)


def test_stale_price_flagged():
    df = _daily(np.array([100.0, 100.0, 100.0, 100.0, 105.0]))
    assert any(f.kind == "stale_price" for f in quality.check_daily_series("BTC", df, stale_days=4))


def test_zero_volume_flagged():
    df = _daily(np.array([100.0, 101.0, 102.0]), volume=[5.0, 0.0, 3.0])
    assert any(f.kind == "zero_volume" for f in quality.check_daily_series("BTC", df))


def test_source_divergence_flagged():
    dates = pd.date_range("2021-01-01", periods=4, freq="D")
    a = pd.DataFrame({"date": dates, "close": [100.0, 101.0, 102.0, 103.0]})
    b = a.copy()
    b.loc[2, "close"] = 108.0  # ~5.7% higher on one day
    flags = quality.check_source_divergence("BTC", a, b, threshold=0.02)
    assert len(flags) == 1 and flags[0].date == "2021-01-03"


def test_intraday_coverage_flagged():
    rdf = pd.DataFrame({"date": pd.date_range("2021-01-01", periods=2), "n_bars": [288, 100]})
    flags = quality.check_intraday_coverage("BTC", rdf)
    assert len(flags) == 1 and flags[0].kind == "few_intraday_bars"


def test_check_all_returns_frame():
    df = _daily(np.array([100.0, 100.0, 160.0, 160.0]))
    out = quality.check_all({"BTC": df})
    assert list(out.columns) == ["kind", "asset", "date", "detail"]
    assert len(out) >= 1
