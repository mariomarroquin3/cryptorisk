"""Known-answer tests for the Acerbi-Szekely ES backtests."""

import numpy as np
from scipy import stats

from cryptorisk.backtest.es_tests import pvalue_by_simulation, z1, z2

A = 0.025


def _normal_var_es(sigma, n):
    v = sigma * stats.norm.ppf(A)
    e = sigma * (-stats.norm.pdf(stats.norm.ppf(A)) / A)
    return np.full(n, v), np.full(n, e)


def test_z2_near_zero_when_es_is_correct():
    rng = np.random.default_rng(0)
    n = 20000
    r = rng.normal(0.0, 0.02, n)
    v, e = _normal_var_es(0.02, n)
    assert abs(z2(r, v, e, A)) < 0.1


def test_z2_negative_when_tails_are_fatter_than_forecast():
    rng = np.random.default_rng(1)
    n = 20000
    r = rng.standard_t(3, n) * 0.02          # much fatter tails
    v, e = _normal_var_es(0.02, n)            # ES from a thin normal
    assert z2(r, v, e, A) < -0.2


def test_z1_nan_without_violations():
    v = np.full(100, -0.5)
    e = np.full(100, -0.6)
    r = np.zeros(100)                         # nothing breaches -0.5
    assert np.isnan(z1(r, v, e))


def test_pvalue_by_simulation_rejects_fat_tails():
    rng = np.random.default_rng(2)
    n = 3000
    r = rng.standard_t(3, n) * 0.02
    v, e = _normal_var_es(0.02, n)
    stat = z2(r, v, e, A)
    sims = rng.normal(0.0, 0.02, size=(400, n))   # H0 world: thin normal
    p = pvalue_by_simulation(stat, sims, v, e, A, which="z2")
    assert p < 0.05
