"""Phase 5 extension: estimation-risk bands on VaR / ES."""

from __future__ import annotations

import warnings

import numpy as np

from cryptorisk.decision.estimation_risk import (
    EstimationRiskBand,
    capital_addon,
    fhs_band,
    garch_t_band,
    hs_band,
)

_A = 0.025


def _garch_series(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h = np.empty(n)
    r = np.empty(n)
    h[0] = 4e-4
    for t in range(n):
        if t:
            h[t] = 3e-6 + 0.07 * r[t - 1] ** 2 + 0.9 * h[t - 1]
        r[t] = rng.standard_t(6) * np.sqrt(h[t]) * np.sqrt(4 / 6)
    return r


# --------------------------------------------------------------------------- #
def test_hs_band_brackets_point_and_prudent_is_conservative():
    r = _garch_series(600, seed=1)
    b = hs_band(r, [_A], n_boot=1500, seed=0)[0]
    assert b.estimator == "HS"
    assert b.es_se > 0
    assert b.var_lo <= b.var_point <= b.var_hi + 1e-9
    # prudent (5th pct of ES draws) is at least as negative as the point ES
    assert b.es_prudent <= b.es_point + 1e-9
    assert b.es_widening() >= -1e-9


def test_garch_t_band_is_well_formed_and_comparable_to_hs():
    r = _garch_series(700, seed=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = garch_t_band(r, [_A], n_draws=1200, seed=0)[0]
    h = hs_band(r, [_A], n_boot=1500, seed=0)[0]
    assert g.n_draws > 100
    assert np.isfinite(g.es_se) and g.es_se > 0
    assert g.var_lo <= g.var_point <= g.var_hi + 1e-9
    assert g.es_prudent <= g.es_point + 1e-9
    # same ballpark as the non-parametric band (guards against a blown-up draw)
    assert 0.2 < g.es_se / h.es_se < 5.0


def test_fhs_band_runs_and_is_finite():
    r = _garch_series(700, seed=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = fhs_band(r, [_A], n_draws=1000, seed=0)[0]
    assert f.estimator == "FHS"
    assert f.n_draws > 100
    assert np.isfinite([f.es_point, f.es_se, f.es_prudent]).all()
    assert f.es_prudent <= f.es_point + 1e-9


def test_shorter_window_has_more_estimation_risk():
    long = hs_band(_garch_series(1500, seed=4), [_A], n_boot=1500, seed=0)[0]
    short = hs_band(_garch_series(250, seed=4), [_A], n_boot=1500, seed=0)[0]
    assert short.es_se > long.es_se


def test_capital_addon_non_negative_and_zero_when_no_widening():
    r = _garch_series(600, seed=5)
    b = hs_band(r, [_A], n_boot=1200, seed=0)[0]
    add = capital_addon(b, liquidity_horizon=10, multiplier=1.5, notional=1_000_000)
    assert add >= 0.0
    # a band whose prudent == point yields exactly zero
    flat = EstimationRiskBand("x", _A, 100, -0.05, -0.06, -0.06, -0.06,
                              0.0, 0.0, -0.06, -0.06, -0.06, -0.06, -0.06)
    assert capital_addon(flat, liquidity_horizon=10, multiplier=1.5, notional=1e6) == 0.0


def test_band_handles_a_too_short_series():
    b = garch_t_band(np.random.default_rng(6).normal(0, 0.02, 40), [_A], seed=0)[0]
    assert b.n_draws == 0
    assert not np.isfinite(b.es_se)
    assert np.isnan(capital_addon(b, liquidity_horizon=10, multiplier=1.5, notional=1e6))


def test_rf_qr_band_is_ordered_and_tight():
    from cryptorisk.decision.estimation_risk import rf_qr_band

    r = _garch_series(600, seed=2)
    b = rf_qr_band(r, [_A], n_draws=200, seed=0)[0]
    assert b.estimator == "RF-QR"
    assert b.es_se >= 0
    assert b.var_point < 0 and b.es_prudent <= b.es_point + 1e-9
    # tree resampling only sees forest Monte Carlo noise: far tighter than an HS bootstrap
    hs = hs_band(r, [_A], n_boot=500, seed=0)[0]
    assert b.es_se < hs.es_se


def test_lstm_band_nan_below_min_draws_and_finite_with_enough():
    import importlib.util

    import pytest

    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed")
    from cryptorisk.decision.estimation_risk import lstm_band

    r = _garch_series(300, seed=3)
    few = lstm_band(r, [_A], n_draws=3, seed=0)[0]
    assert np.isnan(few.es_prudent)
    ok = lstm_band(r, [_A], n_draws=20, seed=0)[0]
    assert ok.estimator == "LSTM-Vol" and np.isfinite(ok.es_prudent)
