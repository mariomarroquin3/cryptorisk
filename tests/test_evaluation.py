"""Phase 3 orchestration: the evaluation batteries in
``cryptorisk.study.run_evaluation`` and ``cryptorisk.study.vol_forecast_eval``.

Synthetic-panel known-answer tests plus a real-data smoke that is skipped when
``data/results/backtests.parquet`` has not been built.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from cryptorisk.backtest.es_tests import z1_pvalue_asymptotic, z2_pvalue_asymptotic
from cryptorisk.config import load_config, repo_root
from cryptorisk.study.run_evaluation import (
    evaluate_coverage,
    evaluate_density,
    evaluate_es,
    evaluate_fz0_mcs,
)
from cryptorisk.study.vol_forecast_eval import evaluate_vol_forecasts

MCS_CFG = {"confidence": 0.90, "block_bootstrap_len": 20, "n_boot": 800}


def _panel(n=3000, seed=7):
    """A 3-model panel over a time-varying latent variance path. 'good' is
    calibrated (normal VaR/ES on the true sigma), 'tight' is 40% too shallow,
    'wide' 60% too deep. One asset, one window, both alphas. The latent daily
    variance is carried in ``_latent_var`` for the vol-forecast test."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-05-16", periods=n)
    lat = 0.03**2 * rng.lognormal(0.0, 0.35, n)  # latent daily variance
    r = rng.normal(0.0, np.sqrt(lat))
    rows = []
    for name, kv in [("good", 1.0), ("tight", 0.6), ("wide", 1.6)]:
        s2 = lat * kv**2
        sig = np.sqrt(s2)
        for a in (0.025, 0.01):
            za, ea = stats.norm.ppf(a), -stats.norm.pdf(stats.norm.ppf(a)) / a
            va = sig * za
            rows.append(
                pd.DataFrame(
                    {
                        "date": dates,
                        "asset": "BTC",
                        "model": name,
                        "window": 500,
                        "alpha": a,
                        "var": va,
                        "es": sig * ea,
                        "sigma2": s2,
                        "realized": r,
                        "violation": r < va,
                        "pit": np.clip(stats.norm.cdf(r / sig), 1e-6, 1 - 1e-6),
                        "_latent_var": lat,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="module")
def panel():
    return _panel()


def test_coverage_flags_the_miscalibrated_models(panel):
    cov = evaluate_coverage(panel, ["BTC"], level=0.05, dq_lags=4)
    at1 = cov[(cov.alpha == 0.01)].set_index("model")
    at25 = cov[(cov.alpha == 0.025)].set_index("model")
    # calibrated model: empirical breach rate close to alpha, passes everything
    assert abs(at1.loc["good", "hit_rate"] - 0.01) < 0.006
    assert at1.loc["good", "passes_all"]
    assert at1.loc["good", "basel_zone"] == "green"
    # 40%-too-shallow VaR: far too many breaches, Kupiec rejects at both alphas
    assert at1.loc["tight", "kupiec_reject"]
    assert at25.loc["tight", "kupiec_reject"]
    assert at1.loc["tight", "basel_zone"] == "red"


def test_coverage_treats_nan_violation_as_no_breach():
    """A day the engine could not form a finite VaR is recorded as `violation
    = NaN` (see backtest/engine.py); it must not be upcast to True by a naive
    `.to_numpy(bool)` and counted as a breach."""
    n = 200
    dates = pd.bdate_range("2019-05-16", periods=n)
    bt = pd.DataFrame({
        "date": dates, "asset": "BTC", "model": "m", "window": 500, "alpha": 0.025,
        "var": -0.02, "es": -0.03, "realized": 0.0, "violation": False, "pit": 0.5,
    })
    # a non-finite-VaR day (as the engine would record it): object dtype so it
    # can hold True/False/NaN, same as the real backtests.parquet column.
    bt["violation"] = bt["violation"].astype(object)
    bt.loc[bt.index[0], "violation"] = np.nan
    cov = evaluate_coverage(bt, ["BTC"], level=0.05, dq_lags=4)
    row = cov.iloc[0]
    assert row["n"] == n
    assert row["hit_rate"] == pytest.approx(0.0)  # not 1/n from the NaN row


def test_es_battery_direction(panel):
    es = evaluate_es(panel, ["BTC"]).set_index(["model", "alpha"])
    # 'tight' ES is optimistic -> Z2 < 0 and the approx p-value rejects
    assert es.loc[("tight", 0.025), "z2"] < 0
    assert es.loc[("tight", 0.025), "es_reject_approx"]
    # 'good' is calibrated -> Z2 near 0, not rejected
    assert abs(es.loc[("good", 0.025), "z2"]) < 0.25
    assert not es.loc[("good", 0.025), "es_reject_approx"]


def test_fz0_mcs_headline_ranks_good_first(panel):
    fz0 = evaluate_fz0_mcs(panel, ["BTC"], MCS_CFG, seed=1)
    a025 = fz0[fz0.alpha == 0.025].set_index("model")
    assert a025.loc["good", "is_best"]
    assert a025.loc["good", "fz0_rank"] == 1
    assert a025.loc["good", "in_mcs"]
    assert not a025.loc["tight", "in_mcs"]  # significantly worse
    # DM vs the best model: 'tight' is worse with a small p-value
    assert a025.loc["tight", "dm_vs_best_p"] < 0.05


def test_density_berkowitz(panel):
    dens = evaluate_density(panel, ["BTC"]).set_index("model")
    assert not dens.loc["good", "reject"]
    assert dens.loc["tight", "reject"]  # PIT too dispersed


def test_vol_forecast_mz_and_mcs(panel):
    rng = np.random.default_rng(2)
    lat = panel.drop_duplicates("date").set_index("date")["_latent_var"]
    # RV = latent variance times multiplicative measurement noise
    rv = pd.DataFrame(
        {
            "asset": "BTC",
            "date": lat.index,
            "rv": lat.to_numpy() * rng.lognormal(0.0, 0.3, lat.size),
        }
    )
    vf = evaluate_vol_forecasts(
        panel, rv, asset="BTC", window=500, mcs_cfg=MCS_CFG, seed=3
    ).set_index("model")
    assert vf.loc["good", "qlike_rank"] == 1
    assert vf.loc["good", "in_mcs_qlike"]
    assert 0.7 < vf.loc["good", "mz_b"] < 1.3  # roughly calibrated slope


def test_asymptotic_es_pvalues_are_probabilities():
    rng = np.random.default_rng(4)
    r = rng.normal(0, 0.02, 4000)
    v = np.full_like(r, 0.02 * stats.norm.ppf(0.025))
    e = np.full_like(r, -0.02 * stats.norm.pdf(stats.norm.ppf(0.025)) / 0.025)
    for p in (z1_pvalue_asymptotic(r, v, e), z2_pvalue_asymptotic(r, v, e, 0.025)):
        assert 0.0 <= p <= 1.0


def test_asymptotic_es_pvalues_discriminate():
    """Both p-values must react to ES misspecification: near 0.5 when calibrated,
    small when ES is too optimistic, large when too conservative."""
    rng = np.random.default_rng(5)
    nu, a, sig, n = 6.0, 0.025, 0.03, 6000
    s = np.sqrt((nu - 2) / nu)
    r = sig * rng.standard_t(nu, n) * s
    q = stats.t.ppf(a, nu)
    v = np.full(n, sig * q * s)
    es_z = -(stats.t.pdf(q, nu) / a) * ((nu + q**2) / (nu - 1.0)) * s
    e = np.full(n, sig * es_z)

    for pfn in (
        lambda vv, ee: z1_pvalue_asymptotic(r, vv, ee),
        lambda vv, ee: z2_pvalue_asymptotic(r, vv, ee, a),
    ):
        assert 0.2 < pfn(v, e) < 0.8            # calibrated
        assert pfn(v, e * 0.7) < 0.05           # ES too optimistic -> reject
        assert pfn(v, e * 1.4) > 0.95           # ES too conservative


# --------------------------------------------------------------------------- #
# real-data smoke
# --------------------------------------------------------------------------- #
_PARQUET = repo_root() / load_config()["paths"]["results"] / "backtests.parquet"


@pytest.mark.skipif(not _PARQUET.exists(), reason="run `make backtest` first")
def test_real_backtests_headline_is_sane():
    bt = pd.read_parquet(_PARQUET)
    bt["date"] = pd.to_datetime(bt["date"])
    fz0 = evaluate_fz0_mcs(bt, ["BTC"], MCS_CFG, seed=load_config()["seed"])
    g = fz0[(fz0.asset == "BTC") & (fz0.alpha == 0.025)]
    assert g["is_best"].sum() == 1
    assert g["in_mcs"].any()
    assert (g["fz0_rank"] == 1).sum() == 1
    assert g.loc[g["is_best"], "in_mcs"].all()
