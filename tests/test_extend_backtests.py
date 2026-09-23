import numpy as np
import pandas as pd

from cryptorisk.models.registry import all_models
from cryptorisk.study.extend_backtests import _job


def _frame(n: int = 620) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n),
        "log_return": rng.standard_t(5, n) * 0.02,
    })


def test_job_returns_only_days_from_start_and_matches_full_run():
    from cryptorisk.backtest.engine import walk_forward

    hs = next(m for m in all_models() if m.name == "HS")
    df = _frame()
    start = df["date"].iloc[-5]
    part = _job(df, hs, "BTC", 500, [0.025, 0.01], start, 1)
    assert part["date"].min() == start and part["date"].nunique() == 5
    assert part["window"].dtype == "int64"
    full = walk_forward(df, hs, alphas=[0.025, 0.01], asset="BTC", window=500).frame
    ref = full[full["date"] >= start].reset_index(drop=True)
    assert np.allclose(part["var"].to_numpy(), ref["var"].to_numpy())


def test_job_empty_when_no_new_days():
    hs = next(m for m in all_models() if m.name == "HS")
    df = _frame()
    out = _job(df, hs, "BTC", 500, [0.025], df["date"].iloc[-1] + pd.Timedelta(days=1), 1)
    assert out.empty


def test_load_backtests_live_prefers_live_rows(monkeypatch, tmp_path):
    from cryptorisk.api import data as D

    key = {"model": "HS", "asset": "BTC", "window": 500, "alpha": 0.025}
    frozen = pd.DataFrame([{**key, "date": pd.Timestamp("2026-09-10"), "var": -0.04}])
    live = pd.DataFrame([
        {**key, "date": pd.Timestamp("2026-09-10"), "var": -0.05},
        {**key, "date": pd.Timestamp("2026-09-11"), "var": -0.06},
    ])
    live.to_parquet(tmp_path / "backtests_live.parquet")
    monkeypatch.setattr(D, "_RESULTS", tmp_path)
    monkeypatch.setattr(D, "load_backtests", lambda: frozen)
    D.load_backtests_live.cache_clear()
    out = D.load_backtests_live().sort_values("date")
    assert list(out["var"]) == [-0.05, -0.06]
    D.load_backtests_live.cache_clear()
