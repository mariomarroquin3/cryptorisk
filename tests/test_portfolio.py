"""Phase 7: portfolio copula VaR / ES.

Known-answer tests for the copula and marginal helpers, plus an artefact-gated
smoke of the ``run_portfolio`` walk-forward on a short OOS window.
"""

from __future__ import annotations

import numpy as np
import pytest

from cryptorisk.config import load_config, repo_root
from cryptorisk.portfolio.copula_var import (
    CopulaVaR,
    _fit_copula,
    _pseudo_obs,
    _sample_copula,
)
from cryptorisk.portfolio.marginal import fit_marginal

_DB = repo_root() / load_config()["paths"]["store"]


def _dependent(n: int, rho_ish: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    common = rng.standard_t(5, n)
    a = rho_ish * common + (1 - rho_ish) * rng.standard_t(5, n)
    b = rho_ish * common + (1 - rho_ish) * rng.standard_t(5, n)
    return np.column_stack([_pseudo_obs(a), _pseudo_obs(b)])


# --------------------------------------------------------------------------- #
def test_pseudo_obs_in_unit_interval_and_monotone():
    z = np.array([3.0, -1.0, 0.0, 5.0, -9.0])
    u = _pseudo_obs(z)
    assert u.min() > 0 and u.max() < 1
    # ranks preserved
    assert np.argsort(z).tolist() == np.argsort(u).tolist()


@pytest.mark.parametrize("fam", ["independence", "gaussian", "student_t", "clayton"])
def test_each_copula_fits_and_samples(fam):
    u = _dependent(1000, 0.8, seed=0)
    used, params = _fit_copula(u, fam, t_df=5.0)
    s = _sample_copula(used, params, 4000, np.random.default_rng(1))
    assert s.shape == (4000, 2)
    assert (s > 0).all() and (s < 1).all()
    if fam == "gaussian":
        assert params["corr"][0, 1] > 0.5
        assert np.corrcoef(s.T)[0, 1] > 0.4
    if fam == "clayton":
        assert params["theta"] > 0.5


@pytest.mark.parametrize("fam", ["independence", "gaussian", "student_t", "clayton"])
def test_each_copula_fits_and_samples_k4(fam):
    rng = np.random.default_rng(11)
    common = rng.standard_t(5, 1500)
    cols = [_pseudo_obs(0.7 * common + 0.3 * rng.standard_t(5, 1500)) for _ in range(4)]
    u = np.column_stack(cols)
    used, params = _fit_copula(u, fam, t_df=5.0, k_dim=4)
    assert params["k_dim"] == 4
    s = _sample_copula(used, params, 5000, np.random.default_rng(12))
    assert s.shape == (5000, 4)
    assert (s > 0).all() and (s < 1).all()
    if fam in ("gaussian", "student_t") and used == fam:
        c = params["corr"]
        assert c.shape == (4, 4)
        assert np.allclose(np.diag(c), 1.0)
        assert np.all(np.linalg.eigvalsh(c) > -1e-8)          # PSD
        assert np.corrcoef(s.T)[0, 1] > 0.3                    # dependence carried through


def test_independence_copula_has_zero_dependence():
    u = _dependent(2000, 0.8, seed=2)
    _, p = _fit_copula(u, "independence", t_df=5.0)
    s = _sample_copula("independence", p, 5000, np.random.default_rng(3))
    assert abs(np.corrcoef(s.T)[0, 1]) < 0.05


def test_clayton_falls_back_to_independence_on_no_lower_tail_dependence():
    rng = np.random.default_rng(4)
    u = np.column_stack([rng.random(1500), rng.random(1500)])   # independent
    used, _ = _fit_copula(u, "clayton", t_df=5.0)
    assert used == "independence"


def test_marginal_fit_on_garch_like_series():
    rng = np.random.default_rng(5)
    n = 700
    h = np.empty(n)
    h[0] = 4e-4
    r = np.empty(n)
    for t in range(n):
        if t:
            h[t] = 5e-6 + 0.08 * r[t - 1] ** 2 + 0.9 * h[t - 1]
        r[t] = rng.normal(0, np.sqrt(h[t]))
    fit = fit_marginal(r)
    assert fit.sigma_next > 0 and np.isfinite(fit.sigma_next)
    assert 0.2 * r.std() < fit.sigma_next < 5 * r.std()
    assert fit.z_resid.size > 100


def test_marginal_fallback_on_short_series():
    fit = fit_marginal(np.random.default_rng(6).normal(0, 0.02, 40))
    assert not fit.ok
    assert np.isinf(fit.nu)


def test_copula_var_invariants_and_diversification():
    rng = np.random.default_rng(7)
    common = rng.standard_normal(600)
    win = {
        "BTC": 0.03 * (0.85 * common + 0.15 * rng.standard_normal(600)),
        "ETH": 0.04 * (0.85 * common + 0.15 * rng.standard_normal(600)),
    }
    w = {"BTC": 0.5, "ETH": 0.5}
    out = {}
    for fam in ("independence", "gaussian", "clayton"):
        fc = CopulaVaR(fam, ["BTC", "ETH"], w, n_sim=15000, seed=1).fit_predict(win, [0.025, 0.01])
        assert fc.var[0.025] < 0
        assert fc.es[0.025] <= fc.var[0.025] + 1e-9
        assert fc.var[0.01] <= fc.var[0.025] + 1e-9
        out[fam] = fc.var[0.025]
    # positively-dependent assets: pretending independence understates basket risk
    assert out["independence"] > out["gaussian"] - 1e-3     # less negative (tighter)


# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _DB.exists(), reason="needs the store")
def test_run_portfolio_short_oos_smoke():
    import pandas as pd

    from cryptorisk.study.run_portfolio import (
        _copula_walk_forward,
        _default_portfolio_cfg,
        _direct_walk_forward,
        _evaluate,
        _returns,
    )

    cfg = load_config()
    cfg = {**cfg, "sample": {**cfg["sample"], "oos_start": "2026-05-01"}}
    pcfg = _default_portfolio_cfg(cfg)
    pcfg = {**pcfg, "n_sim": 3000, "copulas": ["independence", "gaussian"],
            "direct_models": ["HS"], "refit_every": 10}

    wide = _returns(cfg, pcfg["assets"])
    cop = _copula_walk_forward(wide, pcfg, cfg)
    direct = _direct_walk_forward(wide, pcfg, cfg)
    bt = pd.concat([cop, direct], ignore_index=True)
    assert set(bt["model"]) == {"Copula-independence", "Copula-gaussian", "Direct-HS"}
    assert (bt["asset"] == "PORTFOLIO").all()
    assert (bt.loc[bt["var"].notna(), "es"] <= bt.loc[bt["var"].notna(), "var"] + 1e-9).all()

    ev, _sub = _evaluate(bt, cfg)
    assert {"fz0_rank", "in_mcs", "hit_rate", "z2", "berkowitz_p"} <= set(ev.columns)
    assert ev["fz0_rank"].notna().all()
