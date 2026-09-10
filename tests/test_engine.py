"""Walk-forward engine: shape, OOS start, calibration on a synthetic series."""

import numpy as np
import pandas as pd
import pytest

from cryptorisk.backtest.engine import walk_forward
from cryptorisk.models.ewma import EWMA
from cryptorisk.models.historical import HistoricalSimulation

ALPHAS = [0.025, 0.01]


@pytest.fixture(scope="module")
def synth() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 900
    # mild GARCH-like clustering so EWMA has something to track
    s = np.empty(n)
    s[0] = 0.02
    e = rng.standard_normal(n)
    for t in range(1, n):
        s[t] = np.sqrt(0.00002 + 0.05 * (s[t - 1] * e[t - 1]) ** 2 + 0.90 * s[t - 1] ** 2)
    r = s * rng.standard_t(6, size=n)
    dates = pd.date_range("2019-01-01", periods=n, freq="D")
    return pd.DataFrame({"date": dates, "log_return": r})


def test_engine_output_shape_and_columns(synth):
    res = walk_forward(synth, HistoricalSimulation(), alphas=ALPHAS, window=500)
    f = res.frame
    assert set(f.columns) == {
        "date", "asset", "model", "window", "alpha", "var", "es",
        "sigma2", "realized", "violation", "pit",
    }
    n_days = len(synth) - 500
    assert len(f) == n_days * len(ALPHAS)
    assert set(f["alpha"].unique()) == set(ALPHAS)
    assert (f["es"] <= f["var"] + 1e-9).all()


def test_oos_start_is_respected(synth):
    res = walk_forward(synth, HistoricalSimulation(), alphas=[0.025], window=500,
                       oos_start="2020-06-01")
    assert res.frame["date"].min() >= pd.Timestamp("2020-06-01")


def test_hs_is_roughly_calibrated(synth):
    res = walk_forward(synth, HistoricalSimulation(), alphas=[0.025], window=500)
    hr = res.hit_rate(0.025)
    assert 0.01 < hr < 0.06  # HS on ~400 OOS days: near nominal, wide band


def test_ewma_runs_and_has_sigma2(synth):
    res = walk_forward(synth, EWMA(), alphas=ALPHAS, window=500)
    assert res.frame["sigma2"].gt(0).mean() > 0.95


def test_engine_rejects_short_series():
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=100), "log_return": np.zeros(100)})
    with pytest.raises(ValueError):
        walk_forward(df, HistoricalSimulation(), alphas=ALPHAS, window=500)
