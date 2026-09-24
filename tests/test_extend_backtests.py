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


def test_whatif_shock_raises_reactive_models_and_leaves_lstm_cache_alone(monkeypatch):
    from cryptorisk.api import data as D

    rng = np.random.default_rng(1)
    n = 560
    r = rng.standard_t(6, n) * 0.02
    rv = r**2 + 1e-5
    win = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n), "close": 100 * np.exp(np.cumsum(r)),
        "log_return": r, "rv": rv, "bv": rv * 0.9, "rsv_pos": rv / 2, "rsv_neg": rv / 2,
        "jump": 0.0, "rq": rv**2,
    })
    monkeypatch.setattr(D, "load_price_window", lambda asset, n=900: win.tail(n))
    D.whatif_forecast.cache_clear()
    calm = D.whatif_forecast("BTC", "GARCH-t", 0.0)
    crash = D.whatif_forecast("BTC", "GARCH-t", -0.15)
    assert crash["var_0.025"] < calm["var_0.025"] < 0
    rv_crash = D._shock_day_realized(win, -0.15)
    assert rv_crash["rv"] == 0.15**2 and rv_crash["rsv_neg"] == rv_crash["rv"] and rv_crash["rsv_pos"] == 0.0
    shared = D._model_map()["LSTM-Vol"]
    before = dict(getattr(shared, "_cache", {}))
    D.whatif_forecast("BTC", "HAR-RV", -0.1)
    assert dict(getattr(shared, "_cache", {})) == before
    D.whatif_forecast.cache_clear()


def test_intraday_is_skipped_when_price_history_is_stale(monkeypatch):
    from cryptorisk.api import app as A
    from cryptorisk.api import data as D

    called = []
    monkeypatch.setattr(
        D, "intraday_status",
        lambda asset, ref: called.append(ref) or {"low_ret": -0.01, "ret_so_far": 0.0},
    )
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    stale = A._intraday("BTC", today - pd.Timedelta(days=4), 100.0, -0.05, -0.07)
    assert stale is None and not called  # a 4-day-old reference close must not produce a "today" verdict
    fresh = A._intraday("BTC", today - pd.Timedelta(days=1), 100.0, -0.05, -0.07)
    assert fresh is not None and fresh["var_breached"] is False
