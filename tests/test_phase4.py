"""Phase 4: Giacomini-White conditional predictive ability, sub-period
re-evaluation, and the MS-GARCH regime-identification analysis.

The GW test is a synthetic known-answer test; the study-level functions are
gated on ``data/results/backtests.parquet`` / the store / the cached MS-GARCH
predictions and are skipped otherwise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cryptorisk.backtest.scoring import giacomini_white
from cryptorisk.config import load_config, repo_root

_RES = repo_root() / load_config()["paths"]["results"]
_PARQUET = _RES / "backtests.parquet"
_DB = repo_root() / load_config()["paths"]["store"]


# --------------------------------------------------------------------------- #
# Giacomini-White
# --------------------------------------------------------------------------- #
def test_gw_detects_state_dependent_accuracy():
    rng = np.random.default_rng(0)
    n = 2000
    state = rng.normal(size=n)  # the conditioning signal
    h = np.column_stack([np.ones(n), state])
    # A beats B by more when `state` is high: E[d | state] = -0.3 * state
    d = -0.3 * state + rng.normal(0, 0.5, n)
    loss_b = rng.normal(1.0, 0.3, n) ** 2
    loss_a = loss_b + d
    res = giacomini_white(loss_a, loss_b, h)
    assert res.df == 2
    assert res.p_value < 1e-3
    assert res.rejects()
    assert res.favors_high == "A"  # A better when state high


def test_gw_no_rejection_when_equally_accurate():
    rng = np.random.default_rng(1)
    n = 2000
    h = np.column_stack([np.ones(n), rng.normal(size=n)])
    loss_b = rng.normal(1.0, 0.3, n) ** 2
    loss_a = loss_b + rng.normal(0, 0.4, n)  # zero-mean, state-independent
    res = giacomini_white(loss_a, loss_b, h)
    assert res.p_value > 0.10
    assert not res.rejects()


def test_gw_handles_degenerate_input():
    n = 50
    h = np.column_stack([np.ones(n), np.zeros(n)])
    res = giacomini_white(np.ones(n), np.ones(n), h)  # zero variance
    assert not np.isfinite(res.statistic) or not res.rejects()


# --------------------------------------------------------------------------- #
# study-level, artefact-gated
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _PARQUET.exists(), reason="run `make backtest` first")
def test_subperiods_full_oos_matches_and_windows_present():
    from cryptorisk.study.subperiods import evaluate_subperiods

    cfg = load_config()
    bt = pd.read_parquet(_PARQUET)
    bt["date"] = pd.to_datetime(bt["date"])
    sub = evaluate_subperiods(bt, ["BTC"], cfg)
    periods = set(sub["period"])
    assert "full_oos" in periods
    assert periods & {"covid", "luna", "ftx", "calm_23"}
    # every (period, alpha) cell has exactly one FZ0 winner
    for _, g in sub.groupby(["period", "alpha"], observed=True):
        assert g["is_best"].sum() == 1
        assert (g["fz0_rank"] == 1).sum() == 1


@pytest.mark.skipif(not (_PARQUET.exists() and _DB.exists()), reason="needs parquet + store")
def test_gw_cpa_table_is_wellformed():
    from cryptorisk.study.subperiods import evaluate_gw_cpa

    cfg = load_config()
    bt = pd.read_parquet(_PARQUET)
    bt["date"] = pd.to_datetime(bt["date"])
    gw = evaluate_gw_cpa(bt, ["BTC"], cfg)
    assert not gw.empty
    assert gw["gw_p"].dropna().between(0.0, 1.0).all()
    assert gw["rvz_p"].dropna().between(0.0, 1.0).all()
    assert gw["best_edge_vs_rv"].isin(["flat", "shrinks in high RV", "grows in high RV"]).all()


@pytest.mark.skipif(
    not ((_RES / "msgarch_pred_BTC.csv").exists() and _DB.exists()),
    reason="needs cached MS-GARCH predictions + store",
)
def test_regime_identification_insample_beats_walkforward():
    from cryptorisk.study.regime_identification import analyse_cached

    out = analyse_cached()
    for asset in out["asset"].unique():
        g = out[out["asset"] == asset].set_index("series")
        wf = abs(g.loc["filt_wf", "corr_absret"])
        ins = abs(g.loc["insample", "corr_absret"])
        # the whole point: the full-sample fit tracks |r|, the W=500 filter does not
        assert ins > 0.3
        assert wf < 0.2
        assert ins > wf
