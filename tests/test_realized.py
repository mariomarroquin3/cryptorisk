"""Realized measures: known-answer tests on synthetic 5-minute bars."""

import numpy as np
import pandas as pd
import pytest

from cryptorisk.data.realized import clean_bars, realized_daily


def _bars_from_returns(r: np.ndarray, day: str = "2021-06-01", base: float = 30_000.0) -> pd.DataFrame:
    ts = pd.date_range(f"{day} 00:00", periods=len(r) + 1, freq="5min")
    close = base * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
    return pd.DataFrame({"ts": ts, "close": close})


def test_clean_bars_drops_dupe_and_bad_print():
    bars = _bars_from_returns(np.full(50, 0.0005))
    bars = pd.concat([bars, bars.iloc[[10]]], ignore_index=True)          # duplicate ts
    bars.loc[20, "close"] = bars.loc[20, "close"] * 5                     # 400% spike
    bars.loc[25, "close"] = -1.0                                         # non-positive
    cleaned = clean_bars(bars)
    assert cleaned["ts"].is_unique
    assert (cleaned["close"] > 0).all()
    assert len(cleaned) < len(bars)


def test_rv_equals_sum_of_squares_without_subsample():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0, 0.0015, size=280)
    out = realized_daily(_bars_from_returns(r), subsample=False)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["rv"] == pytest.approx(np.sum(r**2), rel=1e-9)
    assert row["n_bars"] == 280
    assert row["rsv_pos"] + row["rsv_neg"] == pytest.approx(row["rv"], rel=1e-9)


def test_no_jump_when_returns_are_smooth():
    # alternating constant |r| -> BV > RV -> jump clamped to 0
    r = np.array([0.001, -0.001] * 140)
    row = realized_daily(_bars_from_returns(r), subsample=False).iloc[0]
    assert row["jump"] == 0.0


def test_jump_detected_with_a_large_spike():
    rng = np.random.default_rng(1)
    r = rng.normal(0.0, 0.0012, size=280)
    r[140] = 0.15  # a 15% move in five minutes
    row = realized_daily(_bars_from_returns(r), subsample=False).iloc[0]
    assert row["jump"] > 0.0
    assert row["rv"] > row["bv"]


def test_subsampled_rv_is_positive_and_close_to_native():
    rng = np.random.default_rng(2)
    r = rng.normal(0.0, 0.0015, size=288)
    bars = _bars_from_returns(r)
    native = realized_daily(bars, subsample=False).iloc[0]["rv"]
    ss = realized_daily(bars, subsample=True).iloc[0]["rv"]
    assert ss > 0
    assert ss == pytest.approx(native, rel=0.5)  # same order of magnitude
