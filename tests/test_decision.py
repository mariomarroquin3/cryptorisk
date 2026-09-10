"""Phase 5: the decision layer (capital / limits / PLA / hedge).

Pure known-answer tests for the four modules, plus artefact-gated smokes for
the ``cryptorisk.study.run_decision`` tables.
"""

from __future__ import annotations

import numpy as np
import pytest

from cryptorisk.config import load_config, repo_root
from cryptorisk.decision import capital as cap
from cryptorisk.decision import hedge as hg
from cryptorisk.decision import limits as lim
from cryptorisk.decision import pnl_attribution as pla

_RES = repo_root() / load_config()["paths"]["results"]
_PARQUET = _RES / "backtests.parquet"
_DB = repo_root() / load_config()["paths"]["store"]


# --------------------------------------------------------------------------- #
# capital
# --------------------------------------------------------------------------- #
def test_es_capital_closed_form():
    got = cap.es_capital(-0.05, liquidity_horizon=10, multiplier=1.5, notional=1_000_000)
    assert got == pytest.approx(1.5 * 1_000_000 * 0.05 * np.sqrt(10))


def test_es_horizon_bootstrap_matches_sqrt_time_for_iid():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.02, 4000)
    es_1d = float(np.mean(r[r <= np.quantile(r, 0.025)]))
    boot = cap.es_horizon_bootstrap(r, 0.025, 10, n_boot=4000, seed=1)
    # iid -> the 10-day ES is about sqrt(10) times the 1-day ES
    assert boot == pytest.approx(np.sqrt(10) * es_1d, rel=0.15)


def test_basel_multiplier_zones():
    assert cap.basel_multiplier(3) == 1.5
    assert cap.basel_multiplier(6) == pytest.approx(2.0)
    assert cap.basel_multiplier(15) == pytest.approx(2.5)


def test_model_risk_addon_is_capital_spread_over_mcs():
    caps = {"a": 100.0, "b": 150.0, "c": 130.0, "d": 999.0}
    assert cap.model_risk_addon(caps, ["a", "b", "c"]) == 50.0
    assert cap.model_risk_addon(caps, ["a"]) == 0.0


# --------------------------------------------------------------------------- #
# limits
# --------------------------------------------------------------------------- #
def test_max_notional():
    assert lim.max_notional(-0.06, 120_000) == pytest.approx(2_000_000)
    assert lim.max_notional(0.0, 120_000) == float("inf")


def test_limit_backtest_calibrated_series():
    rng = np.random.default_rng(2)
    n = 3000
    sigma = 0.03
    r = rng.normal(0, sigma, n)
    es = np.full(n, sigma * -np.exp(-0.5 * 2.326**2) / (0.01 * np.sqrt(2 * np.pi)))  # 99% normal ES
    n_star = lim.max_notional(float(es.mean()), 120_000)
    bkt = lim.backtest_framework(es, r, n_star, budget=120_000)
    assert bkt.mean_utilisation == pytest.approx(1.0, abs=1e-9)
    assert 0.003 < bkt.es_exceedance_rate < 0.02  # ~1%, sampling slack
    assert bkt.n_budget_breaches >= 1
    assert bkt.worst_loss > 0


# --------------------------------------------------------------------------- #
# PLA
# --------------------------------------------------------------------------- #
def test_pla_identical_series_is_green():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1e6, 500)
    res = pla.pla_test(x, x.copy())
    assert res.spearman == pytest.approx(1.0)
    assert res.ks == pytest.approx(0.0)
    assert res.zone == "green"


def test_pla_wrong_shape_is_not_green():
    rng = np.random.default_rng(4)
    hpl = rng.standard_t(3, 4000) * 1e5  # fat-tailed realised
    rtpl = rng.normal(0, 1e5, 4000)  # thin, and unrelated
    res = pla.pla_test(rtpl, hpl)
    assert res.zone in {"amber", "red"}


def test_implied_rtpl_recovers_hpl_for_calibrated_gaussian():
    rng = np.random.default_rng(5)
    sigma2 = np.full(1000, 0.02**2)
    r = rng.normal(0, 0.02, 1000)
    from scipy import stats

    pit = stats.norm.cdf(r / 0.02)
    rtpl = pla.implied_rtpl(sigma2, pit, 1_000_000)
    assert np.corrcoef(rtpl, 1_000_000 * r)[0, 1] > 0.999


# --------------------------------------------------------------------------- #
# hedge
# --------------------------------------------------------------------------- #
def test_min_variance_ratio_perfect_and_none():
    rng = np.random.default_rng(6)
    s = rng.normal(0, 0.03, 2000)
    assert hg.min_variance_ratio(s, s) == pytest.approx(1.0)
    assert abs(hg.min_variance_ratio(s, rng.normal(0, 0.03, 2000))) < 0.1


def test_es_minimising_ratio_hedges_out_a_shared_factor():
    rng = np.random.default_rng(7)
    perp = rng.normal(0, 0.03, 4000)
    spot = perp + rng.normal(0, 0.005, 4000)  # spot = perp + small idiosyncratic
    h = hg.es_minimising_ratio(spot, perp, 0.025)
    assert 0.8 < h < 1.2


def test_funding_carry_annualised_sign():
    # positive funding -> short-perp hedge earns; annualised = rate * 3 * 365
    assert hg.funding_carry_annualised(np.full(100, 1e-4)) == pytest.approx(1e-4 * 3 * 365)
    assert hg.funding_carry_annualised(np.full(100, -2e-4)) < 0


# --------------------------------------------------------------------------- #
# orchestrator smoke
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (_PARQUET.exists() and _DB.exists()), reason="needs parquet + store")
def test_run_decision_tables_are_wellformed():
    import pandas as pd

    from cryptorisk.study.run_decision import (
        _load,
        _mcs_members,
        capital_table,
        hedge_table,
        limits_table,
        pla_table,
    )

    cfg = load_config()
    bt, ret_oos, fund_oos = _load(cfg)
    mcs = _mcs_members(_RES)

    capdf = capital_table(bt, mcs, cfg)
    assert (capdf["capital_usd"] > 0).all()
    assert (capdf["m_c"] >= cfg["decision"]["basel_multiplier_base"]).all()
    assert (capdf.groupby("asset")["model_risk_addon_usd"].nunique() == 1).all()
    # the traffic-light exception count must come from the 99% (alpha=0.01)
    # backtest, not the 97.5% ES slice used for the capital amount
    from cryptorisk.backtest.coverage import basel_traffic_light

    row = capdf.iloc[0]
    v99 = (
        bt[
            (bt.asset == row["asset"])
            & (bt.window == 500)
            & (bt.alpha == 0.01)
            & (bt.model == row["model"])
        ]
        .sort_values("date")["violation"]
        .to_numpy(bool)
    )
    assert row["exceptions_250d"] == basel_traffic_light(v99).exceptions

    limdf = limits_table(bt, mcs, cfg)
    assert (limdf["n_star_usd"] > 0).all()
    assert limdf["mean_utilisation"].between(0.99, 1.01).all()
    assert limdf["es_exceedance_rate"].between(0.0, 0.05).all()

    pladf = pla_table(bt, mcs, cfg)
    assert pladf["pla_zone"].isin(["green", "amber", "red"]).all()

    hgdf = hedge_table(ret_oos, fund_oos, cfg)
    assert len(hgdf) == len(cfg["assets"])
    assert np.isfinite(hgdf["funding_carry_annual_usd"]).all()
    assert isinstance(pd.DataFrame(hgdf), pd.DataFrame)
