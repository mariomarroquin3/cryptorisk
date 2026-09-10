"""Known-answer tests for the VaR coverage battery."""

import numpy as np
import pytest

from cryptorisk.backtest.coverage import (
    basel_traffic_light,
    christoffersen_cc,
    christoffersen_independence,
    kupiec_pof,
)


def _bernoulli(rate: float, n: int, seed: int = 0) -> np.ndarray:
    return (np.random.default_rng(seed).random(n) < rate).astype(int)


def test_kupiec_hand_value():
    # n=250, x=10, alpha=0.05 -> LR_uc computed by hand.
    v = np.zeros(250, dtype=int)
    v[:10] = 1
    r = kupiec_pof(v, 0.05)
    n, x, a = 250, 10, 0.05
    pi = x / n
    ll0 = (n - x) * np.log(1 - a) + x * np.log(a)
    ll1 = (n - x) * np.log(1 - pi) + x * np.log(pi)
    assert r.statistic == pytest.approx(-2 * (ll0 - ll1), rel=1e-12)
    assert r.df == 1


def test_kupiec_calibrated_series_not_rejected():
    v = _bernoulli(0.05, 4000, seed=1)
    assert not kupiec_pof(v, 0.05).rejects()


def test_kupiec_miscalibrated_series_rejected():
    v = _bernoulli(0.15, 2000, seed=2)  # 3x the nominal rate
    assert kupiec_pof(v, 0.05).rejects()


def test_kupiec_degenerate_is_noninformative():
    r = kupiec_pof(np.zeros(500, dtype=int), 0.05)
    assert np.isnan(r.p_value)
    assert not r.rejects()


def test_independence_iid_not_rejected():
    v = _bernoulli(0.05, 5000, seed=3)
    assert not christoffersen_independence(v).rejects()


def test_independence_clustered_rejected():
    # 40 violations, all consecutive -> strong 1-lag dependence.
    v = np.zeros(1000, dtype=int)
    v[100:140] = 1
    assert christoffersen_independence(v).rejects()


def test_cc_combines_df():
    v = _bernoulli(0.05, 3000, seed=4)
    assert christoffersen_cc(v, 0.05).df == 2


def test_basel_zones():
    def viol(k):
        v = np.zeros(250, dtype=int)
        v[:k] = 1
        return v

    assert basel_traffic_light(viol(4)).zone == "green"
    assert basel_traffic_light(viol(4)).multiplier_addon == 0.0
    amber = basel_traffic_light(viol(5))
    assert amber.zone == "amber" and amber.multiplier_addon == pytest.approx(0.40)
    red = basel_traffic_light(viol(10))
    assert red.zone == "red" and red.multiplier_addon == 1.0


def test_dq_not_rejected_for_calibrated_iid_hits():
    rng = np.random.default_rng(10)
    n = 4000
    var = -0.05 + rng.normal(0, 0.002, n)          # mild variation, unrelated to hits
    viol = (rng.random(n) < 0.025).astype(int)
    from cryptorisk.backtest.coverage import dq_test

    assert not dq_test(viol, var, 0.025).rejects()


def test_dq_rejects_hits_predictable_from_var():
    rng = np.random.default_rng(11)
    n = 4000
    var = -0.03 - 0.05 * rng.random(n)             # wide range of VaR levels
    # violations happen only when VaR is shallow -> hits predictable from VaR_t
    p = np.where(var > -0.05, 0.10, 0.005)
    viol = (rng.random(n) < p).astype(int)
    from cryptorisk.backtest.coverage import dq_test

    assert dq_test(viol, var, 0.025).rejects()


def test_dq_rejects_clustered_hits():
    from cryptorisk.backtest.coverage import dq_test

    n = 2000
    viol = np.zeros(n, dtype=int)
    viol[500:560] = 1                              # a solid block -> strong AR in hits
    var = np.full(n, -0.05)
    assert dq_test(viol, var, 0.025).rejects()


def test_dq_noninformative_without_violations():
    from cryptorisk.backtest.coverage import dq_test

    r = dq_test(np.zeros(500, dtype=int), np.full(500, -0.05), 0.025)
    assert np.isnan(r.p_value) and not r.rejects()
