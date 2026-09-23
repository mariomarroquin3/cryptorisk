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


def test_live_track_record_counts_violations_and_ranks(monkeypatch, tmp_path):
    from cryptorisk.api import data as D

    dates = pd.date_range("2026-09-11", periods=4)
    rows = []
    for model, var in (("A", -0.05), ("B", -0.01)):
        for d, r in zip(dates, (0.01, -0.03, 0.0, 0.02), strict=True):
            rows.append({
                "model": model, "asset": "BTC", "window": 500, "alpha": 0.025, "date": d,
                "var": var, "es": var * 1.3, "realized": r, "violation": r < var,
            })
    pd.DataFrame(rows).to_parquet(tmp_path / "backtests_live.parquet")
    monkeypatch.setattr(D, "_RESULTS", tmp_path)
    D.live_track_record.cache_clear()
    out = D.live_track_record("BTC", 0.025)
    by = {m["model"]: m for m in out["models"]}
    assert len(out["days"]) == 4 and by["A"]["violations"] == 0 and by["B"]["violations"] == 1
    assert by["B"]["min_margin"] < 0 < by["A"]["min_margin"]
    assert 0 < by["B"]["p_at_least"] < 1 and by["A"]["p_at_least"] == 1.0
    assert sorted(m["fz0_rank"] for m in out["models"]) == [1, 2]
    D.live_track_record.cache_clear()
