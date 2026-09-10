"""Known-answer tests for the Berkowitz PIT test."""

import numpy as np
from scipy import stats

from cryptorisk.backtest.pit import berkowitz, pit_values


def test_pit_values_clip_and_drop_nan():
    u = pit_values([0.0, 1.0, 0.5, np.nan, -0.1, 1.2])
    assert u.min() > 0 and u.max() < 1
    assert np.isfinite(u).all()


def test_berkowitz_rejection_rate_near_nominal():
    # over many correct-PIT samples the LR ~ chi2(3): rejection rate ~ 5%
    rng = np.random.default_rng(0)
    rej = sum(berkowitz(rng.uniform(0, 1, 2000)).p_value < 0.05 for _ in range(300))
    assert rej / 300 < 0.12   # comfortably above 0.05, well below "always rejects"


def test_berkowitz_rejects_overdispersed_forecasts():
    # forecasts too wide -> PIT clusters near 0.5 -> z has variance << 1
    z = np.random.default_rng(1).normal(0, 0.4, 3000)
    u = stats.norm.cdf(z)
    assert berkowitz(u).p_value < 0.01


def test_berkowitz_rejects_autocorrelated_pit():
    rng = np.random.default_rng(2)
    z = np.empty(3000)
    z[0] = rng.normal()
    for t in range(1, z.size):
        z[t] = 0.6 * z[t - 1] + rng.normal(0, 0.8)
    u = stats.norm.cdf(z)
    assert berkowitz(u).p_value < 0.01
