"""Sanity tests: every model yields a coherent 1-step distribution."""

import contextlib
import importlib.util

import numpy as np
import pandas as pd
import pytest

from cryptorisk.models.base import Context
from cryptorisk.models.registry import phase2a_models, phase2b_models, phase2d_models

A1, A2 = 0.025, 0.01
_HAS_TORCH = importlib.util.find_spec("torch") is not None
ALL = [
    m
    for m in phase2a_models() + phase2b_models(alphas=(A1, A2)) + phase2d_models()
    if _HAS_TORCH or m.name != "LSTM-Vol"  # torch is the optional `ml` extra
]


@pytest.fixture(scope="module")
def ctx() -> Context:
    rng = np.random.default_rng(7)
    n = 900
    # GARCH-ish vol so the realized block carries signal
    s = np.empty(n)
    s[0] = 0.02
    e = rng.standard_normal(n)
    for t in range(1, n):
        s[t] = np.sqrt(3e-5 + 0.06 * (s[t - 1] * e[t - 1]) ** 2 + 0.9 * s[t - 1] ** 2)
    r = s * rng.standard_t(6, size=n)
    # synthetic realized measures: rv ~ true variance + noise; rq ~ rv^2
    rv = s**2 * rng.uniform(0.7, 1.3, size=n)
    rq = 3.0 * rv**2 * rng.uniform(0.7, 1.3, size=n)
    dates = np.arange(np.datetime64("2018-01-01"), np.datetime64("2018-01-01") + np.timedelta64(n, "D"))
    return Context(returns=r, dates=dates, asof=dates[-1], asset="BTC",
                   realized={"rv": rv, "rq": rq})


@pytest.mark.parametrize("model", ALL, ids=lambda m: m.name)
def test_model_predictive_dist_is_coherent(model, ctx):
    d = model.fit_predict(ctx)

    v1, v2 = d.var(A1), d.var(A2)
    assert np.isfinite(v1) and np.isfinite(v2)
    assert v1 < 0 and v2 < 0
    assert v2 <= v1 + 1e-9                        # 99% at least as extreme as 97.5%
    assert abs(v1) < 0.7                          # sane magnitude for ~2%/day

    e1 = d.es(A1)
    assert np.isfinite(e1) and e1 <= v1 + 1e-9    # ES at least as severe as VaR

    with contextlib.suppress(NotImplementedError):  # CAViaR has no variance
        s2 = d.sigma2()
        assert np.isfinite(s2) and s2 > 0
    with contextlib.suppress(NotImplementedError):   # CAViaR has no cdf
        assert abs(d.cdf(v1) - A1) < 0.02


@pytest.mark.parametrize("model", ALL, ids=lambda m: m.name)
def test_model_is_deterministic(model, ctx):
    assert isinstance(model.name, str) and model.name
    a = model.fit_predict(ctx).var(A1)
    b = model.fit_predict(ctx).var(A1)
    assert a == pytest.approx(b, rel=1e-6)


def test_har_falls_back_without_realized():
    rng = np.random.default_rng(3)
    r = rng.standard_t(6, 400) * 0.02
    dates = pd.date_range("2020-01-01", periods=400).to_numpy()
    c = Context(returns=r, dates=dates, asof=dates[-1])
    from cryptorisk.models.har import HAR

    d = HAR().fit_predict(c)
    assert d.var(A1) < 0  # empirical fallback still coherent


def test_realized_sv_falls_back_without_realized():
    rng = np.random.default_rng(3)
    r = rng.standard_t(6, 400) * 0.02
    dates = pd.date_range("2020-01-01", periods=400).to_numpy()
    c = Context(returns=r, dates=dates, asof=dates[-1])
    from cryptorisk.models.stochastic_vol import RealizedSV

    d = RealizedSV().fit_predict(c)
    assert d.var(A1) < 0  # empirical fallback still coherent


def test_random_forest_qr_runs_without_realized():
    rng = np.random.default_rng(3)
    r = rng.standard_t(6, 400) * 0.02
    dates = pd.date_range("2020-01-01", periods=400).to_numpy()
    c = Context(returns=r, dates=dates, asof=dates[-1])
    from cryptorisk.models.random_forest import RandomForestQR

    d = RandomForestQR().fit_predict(c)
    assert d.var(A1) < 0  # squared-return-only features still coherent


def test_random_forest_qr_falls_back_on_short_window():
    rng = np.random.default_rng(3)
    r = rng.standard_t(6, 60) * 0.02
    dates = pd.date_range("2020-01-01", periods=60).to_numpy()
    c = Context(returns=r, dates=dates, asof=dates[-1])
    from cryptorisk.models.random_forest import RandomForestQR

    d = RandomForestQR().fit_predict(c)
    assert d.var(A1) < 0  # empirical fallback (too few valid feature rows)


@pytest.mark.skipif(not _HAS_TORCH, reason="torch (ml extra) not installed")
def test_lstm_vol_runs_and_is_deterministic():
    rng = np.random.default_rng(5)
    r = rng.standard_t(6, 400) * 0.02
    dates = pd.date_range("2020-01-01", periods=400).to_numpy()
    c = Context(returns=r, dates=dates, asof=dates[-1])
    from cryptorisk.models.lstm_vol import LstmVol

    a = LstmVol().fit_predict(c).var(A1)
    b = LstmVol().fit_predict(c).var(A1)
    assert a < 0
    assert a == pytest.approx(b, rel=1e-6)


def test_lstm_vol_falls_back_on_short_window():
    rng = np.random.default_rng(5)
    r = rng.standard_t(6, 60) * 0.02
    dates = pd.date_range("2020-01-01", periods=60).to_numpy()
    c = Context(returns=r, dates=dates, asof=dates[-1])
    from cryptorisk.models.lstm_vol import LstmVol

    d = LstmVol().fit_predict(c)
    assert d.var(A1) < 0  # empirical fallback (window shorter than _MIN_TRAIN + _SEQ_LEN)


def test_realized_sv_state_stays_nonnegative():
    """The CIR state truncation (full truncation Euler scheme) must clamp the
    filtered variance itself, not just intermediate sqrt/log evaluations --
    a regression this specific test would have caught during development,
    when an unclamped state went as low as -0.89 and then diverged."""
    from cryptorisk.models.stochastic_vol import _filter

    rng = np.random.default_rng(11)
    n = 900
    s = np.empty(n)
    s[0] = 0.02
    e = rng.standard_normal(n)
    for t in range(1, n):
        s[t] = np.sqrt(3e-5 + 0.06 * (s[t - 1] * e[t - 1]) ** 2 + 0.9 * s[t - 1] ** 2)
    r = s * rng.standard_t(6, size=n)
    rv = s**2 * rng.uniform(0.7, 1.3, size=n)
    log_rv = np.log(rv)

    # A deliberately unstable kappa (>= 2 breaks the Euler discretization's
    # stability, |1 - kappa| >= 1) to confirm the state stays sane even in a
    # corner the optimizer's bounds are meant to exclude.
    theta = np.array([r.mean(), 3.0, r.var(), 0.3 * np.sqrt(r.var()), 0.0, np.log(0.3), 0.1, 6.0, 1.0])
    v_pred, v_post, p_last, ll, lev = _filter(theta, r, log_rv)
    assert np.all(np.isfinite(v_pred)) and np.all(v_pred >= 0)
    assert np.all(np.isfinite(v_post)) and np.all(v_post >= 0)
    assert np.isfinite(p_last) and np.isfinite(ll)
    assert np.isfinite(lev) and lev >= 0


def test_random_forest_qr_explain_is_consistent():
    from cryptorisk.models.random_forest import RandomForestQR

    rng = np.random.default_rng(9)
    n = 400
    r = rng.standard_t(6, n) * 0.02
    rv = (r**2) * rng.uniform(0.7, 1.3, n) + 1e-6
    dates = pd.date_range("2020-01-01", periods=n).to_numpy()
    c = Context(returns=r, dates=dates, asof=dates[-1], realized={"rv": rv})
    m = RandomForestQR(n_estimators=50)
    e = m.explain(c)
    assert e is not None
    assert len(e["features"]) == len(e["importance"]) == len(e["zscores"]) == 8
    assert sum(e["importance"]) == pytest.approx(1.0, abs=1e-6)
    assert 1.0 <= e["ess"] <= e["n_train"]
    assert e["var_cond"] == pytest.approx(m.fit_predict(c).var(0.025))
    assert m.explain(Context(returns=r[:60], dates=dates[:60], asof=dates[59])) is None


def test_backtest_window_normalised_for_parquet(tmp_path):
    from cryptorisk.study.run_backtests import EXPANDING_WINDOW, _normalize_window

    a = pd.DataFrame({"window": [500, 500], "x": [1.0, 2.0]})
    b = pd.DataFrame({"window": ["expanding"], "x": [3.0]})
    out = _normalize_window(a)
    out = pd.concat([out, _normalize_window(b)], ignore_index=True)
    assert list(out["window"]) == [500, 500, EXPANDING_WINDOW]
    out.to_parquet(tmp_path / "bt.parquet", index=False)  # used to raise ArrowInvalid


@pytest.mark.skipif(not _HAS_TORCH, reason="torch (ml extra) not installed")
def test_lstm_vol_explain_shape_and_features():
    from cryptorisk.models.lstm_vol import LstmVol

    rng = np.random.default_rng(5)
    r = rng.standard_t(6, 300) * 0.02
    dates = pd.date_range("2020-01-01", periods=300).to_numpy()
    c = Context(returns=r, dates=dates, asof=dates[-1])
    e = LstmVol().explain(c, repeats=1)
    imp = np.asarray(e["importance"])
    assert imp.shape == (20, 3) and np.all(np.isfinite(imp))
    assert e["features"] == ["return", "squared return", "squared down-return"]
    assert LstmVol().explain(Context(returns=r[:60], dates=dates[:60], asof=dates[59])) is None
