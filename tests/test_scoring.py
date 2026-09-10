"""Known-answer tests for FZ0 / QLIKE / Diebold-Mariano / Model Confidence Set."""

import numpy as np
import pytest
from scipy import stats

from cryptorisk.backtest.scoring import (
    diebold_mariano,
    fz0_loss,
    model_confidence_set,
    qlike_loss,
)

A = 0.025


def test_fz0_lower_at_true_var_es():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0, 0.02, size=6000)
    v_true = 0.02 * stats.norm.ppf(A)
    e_true = 0.02 * (-stats.norm.pdf(stats.norm.ppf(A)) / A)
    v_bad, e_bad = 0.5 * v_true, 0.5 * e_true  # far too tight
    l_true = fz0_loss(r, np.full_like(r, v_true), np.full_like(r, e_true), A).mean()
    l_bad = fz0_loss(r, np.full_like(r, v_bad), np.full_like(r, e_bad), A).mean()
    assert np.isfinite(l_true) and l_true < l_bad


def test_qlike_zero_at_truth_positive_elsewhere():
    rv = np.abs(np.random.default_rng(1).normal(0, 1, 500)) + 0.1
    assert qlike_loss(rv, rv) == pytest.approx(np.zeros_like(rv), abs=1e-12)
    assert (qlike_loss(2 * rv, rv) > 0).all()


def test_dm_no_significant_difference_for_same_process():
    rng = np.random.default_rng(2)
    a = rng.normal(1.0, 0.2, 1200) ** 2
    b = rng.normal(1.0, 0.2, 1200) ** 2   # same DGP, independent draws
    assert diebold_mariano(a, b).p_value > 0.10


def test_dm_detects_a_better_model():
    rng = np.random.default_rng(3)
    b = rng.normal(1.0, 0.1, size=1500) ** 2
    a = b - 0.3 + rng.normal(0, 0.05, size=1500)   # A lower on average, with noise
    res = diebold_mariano(a, b)
    assert res.mean_diff < 0 and res.favors() == "A" and res.p_value < 0.01


def test_mcs_keeps_the_best_drops_the_worst():
    rng = np.random.default_rng(4)
    T = 1500
    base = rng.normal(1.0, 0.3, T) ** 2
    losses = {
        "best": base,
        "mid": base + 0.15 + rng.normal(0, 0.02, T),
        "worst": base + 0.6 + rng.normal(0, 0.02, T),
    }
    res = model_confidence_set(losses, confidence=0.90, n_boot=1500, seed=1)
    assert res.best == "best"
    assert "best" in res.included
    assert "worst" not in res.included
    assert res.p_values["worst"] < res.p_values["best"]


def test_mcs_equivalent_models_all_included():
    rng = np.random.default_rng(5)
    base = rng.normal(1.0, 0.3, 1200) ** 2
    sd = 0.02 * base.std()
    losses = {k: base + rng.normal(0, sd, base.size) for k in ("a", "b", "c")}
    res = model_confidence_set(losses, confidence=0.90, n_boot=1200, seed=2)
    assert set(res.included) == {"a", "b", "c"}
