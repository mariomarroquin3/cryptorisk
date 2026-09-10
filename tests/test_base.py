"""Known-answer tests for the model interface (Context / PredictiveDist)."""

import numpy as np
import pytest
from scipy import stats

from cryptorisk.models.base import (
    Context,
    EmpiricalDist,
    Model,
    ParametricDist,
    QuantileDist,
)

A = 0.025


def _normal_dist(loc=0.0, scale=1.0, exact_es=True):
    z_es = None
    if exact_es:
        z_es = lambda a: -stats.norm.pdf(stats.norm.ppf(a)) / a  # noqa: E731
    return ParametricDist(loc, scale, stats.norm.ppf, stats.norm.cdf, z_es)


def test_parametric_normal_var_matches_closed_form():
    d = _normal_dist(loc=0.001, scale=0.03)
    assert d.var(A) == pytest.approx(0.001 + 0.03 * stats.norm.ppf(A), rel=1e-12)
    assert d.sigma2() == pytest.approx(0.03**2)


def test_parametric_normal_es_closed_form():
    d = _normal_dist(loc=0.0, scale=1.0, exact_es=True)
    expected = -stats.norm.pdf(stats.norm.ppf(A)) / A
    assert d.es(A) == pytest.approx(expected, rel=1e-10)


def test_parametric_es_quadrature_matches_closed_form():
    d = _normal_dist(exact_es=False)  # numeric integration path
    expected = -stats.norm.pdf(stats.norm.ppf(A)) / A
    assert d.es(A) == pytest.approx(expected, rel=5e-3)


def test_parametric_cdf_ppf_roundtrip():
    d = _normal_dist(loc=0.002, scale=0.04)
    x = 0.01
    assert d.ppf(d.cdf(x)) == pytest.approx(x, rel=1e-9)


def test_parametric_rejects_bad_scale():
    with pytest.raises(ValueError):
        ParametricDist(0.0, -1.0, stats.norm.ppf, stats.norm.cdf)


def test_empirical_matches_numpy():
    rng = np.random.default_rng(0)
    sample = rng.standard_t(5, size=4000) * 0.02
    d = EmpiricalDist(sample)
    q = np.quantile(sample, A)
    # weighted-quantile with equal weights lands within one order statistic
    assert d.var(A) == pytest.approx(q, abs=2e-3)
    assert d.es(A) == pytest.approx(sample[sample <= d.var(A)].mean(), rel=1e-9)


def test_empirical_scale_shift():
    sample = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    d = EmpiricalDist(sample, scale=0.5, loc=0.1)
    base = EmpiricalDist(sample)
    assert d.var(0.2) == pytest.approx(0.1 + 0.5 * base.var(0.2))


def test_quantile_dist_lookup_and_missing():
    d = QuantileDist({0.025: -0.05, 0.01: -0.08}, {0.025: -0.07, 0.01: -0.10})
    assert d.var(0.01) == -0.08
    assert d.es(0.025) == -0.07
    with pytest.raises(KeyError):
        d.var(0.05)
    with pytest.raises(NotImplementedError):
        d.sigma2()


def test_alpha_validation():
    d = _normal_dist()
    for bad in (0.0, 0.5, 0.9, -0.1):
        with pytest.raises(ValueError):
            d.var(bad)


def test_context_validation():
    good = np.array([0.01, -0.02, 0.0, 0.03])
    dates = np.arange("2020-01-01", "2020-01-05", dtype="datetime64[D]")
    ctx = Context(good, dates, asof=dates[-1], asset="BTC")
    assert ctx.n == 4

    with pytest.raises(ValueError):
        Context(np.array([0.01, np.nan]), dates[:2], asof=dates[1])
    with pytest.raises(ValueError):
        Context(good, dates[:3], asof=dates[-1])
    with pytest.raises(ValueError):
        Context(np.array([0.01]), dates[:1], asof=dates[0])


def test_model_protocol_runtime_checkable():
    class Dummy:
        name = "dummy"

        def fit_predict(self, ctx):  # noqa: ARG002
            return _normal_dist()

    assert isinstance(Dummy(), Model)
    assert not isinstance(object(), Model)
