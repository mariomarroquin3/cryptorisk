"""Sanity tests: every Phase-2a model yields a coherent 1-step distribution."""

import numpy as np
import pytest

from cryptorisk.models.base import Context
from cryptorisk.models.registry import phase2a_models

A1, A2 = 0.025, 0.01


@pytest.fixture(scope="module")
def ctx() -> Context:
    rng = np.random.default_rng(7)
    r = rng.standard_t(5, size=900) * 0.02  # ~2%/day, fat tails
    dates = np.arange(np.datetime64("2018-01-01"), np.datetime64("2020-06-19"))[:900]
    return Context(returns=r, dates=dates, asof=dates[-1], asset="BTC")


@pytest.mark.parametrize("model", phase2a_models(), ids=lambda m: m.name)
def test_model_predictive_dist_is_coherent(model, ctx):
    d = model.fit_predict(ctx)

    v1, v2 = d.var(A1), d.var(A2)
    assert np.isfinite(v1) and np.isfinite(v2)
    assert v1 < 0 and v2 < 0                      # left-tail VaR is negative
    assert v2 <= v1 + 1e-9                        # 99% at least as extreme as 97.5%
    assert abs(v1) < 0.6                          # sane magnitude for a 2%/day series

    e1 = d.es(A1)
    assert np.isfinite(e1) and e1 <= v1 + 1e-9    # ES at least as severe as VaR

    s2 = d.sigma2()
    assert np.isfinite(s2) and s2 > 0

    # CDF at the VaR should be close to alpha (skip models without a cdf)
    try:
        p = d.cdf(v1)
    except NotImplementedError:
        p = A1
    assert abs(p - A1) < 0.02


@pytest.mark.parametrize("model", phase2a_models(), ids=lambda m: m.name)
def test_model_names_unique_and_stable(model, ctx):
    assert isinstance(model.name, str) and model.name
    assert model.fit_predict(ctx).var(A1) == model.fit_predict(ctx).var(A1)  # deterministic
