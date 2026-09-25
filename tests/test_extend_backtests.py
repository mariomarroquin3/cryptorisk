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


def _synthetic_window(n=560, last_shock=None, seed=5):
    rng = np.random.default_rng(seed)
    r = rng.standard_t(6, n) * 0.02
    if last_shock is not None:
        r[-1] = last_shock
    rv = r**2 + 1e-5
    return pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n), "close": 100 * np.exp(np.cumsum(r)),
        "log_return": r, "rv": rv, "bv": rv * 0.9, "rsv_pos": rv / 2, "rsv_neg": rv / 2, "jump": 0.0, "rq": rv**2,
    })


def test_var_change_attribution_adds_up_and_blames_the_new_shock(monkeypatch):
    from cryptorisk.api import data as D

    win = _synthetic_window(last_shock=-0.15)
    monkeypatch.setattr(D, "load_price_window", lambda asset, n=900: win.tail(n))
    D.var_change_attribution.cache_clear()
    res = D.var_change_attribution("BTC", "GARCH-t", (0.025,))
    k = "var_0.025"
    total = res["now_" + k] - res["prev_" + k]
    assert abs(res["new_" + k] + res["old_" + k] - total) < 1e-12        # exact decomposition
    assert res["new_" + k] < -0.005                                       # a -15% day deepens the VaR ...
    assert abs(res["old_" + k]) < abs(res["new_" + k]) / 5               # ... far more than an ordinary day leaving
    assert res["new_return"] == win["log_return"].iloc[-1] and res["dropped_date"] == win["date"].iloc[-501]
    D.var_change_attribution.cache_clear()


def test_var_change_endpoint_shapes_and_rejects_offline_models(monkeypatch):
    import pytest
    from fastapi import HTTPException

    from cryptorisk.api import app as A
    from cryptorisk.api import data as D

    win = _synthetic_window()
    monkeypatch.setattr(D, "load_price_window", lambda asset, n=900: win.tail(n))
    D.var_change_attribution.cache_clear()
    out = A.var_change("BTC", "GARCH-t", 0.025)
    assert set(out["var"]) == {"prev", "now", "new", "old"} and out["asof"] > out["prev_asof"]
    assert abs(out["var"]["new"] + out["var"]["old"] - (out["var"]["now"] - out["var"]["prev"])) < 1e-12
    with pytest.raises(HTTPException) as e:
        A.var_change("BTC", "MS-GARCH", 0.025)
    assert e.value.status_code == 404
    D.var_change_attribution.cache_clear()


def test_lstm_local_endpoint_returns_days_oldest_first(monkeypatch):
    from cryptorisk.api import app as A
    from cryptorisk.api import data as D

    rows = [
        {"asset": "BTC", "alpha": 0.025, "asof": "2026-09-23", "date": f"2026-09-{d:02d}", "lag": 24 - d,
         "ret": 0.01 * d, "per_day": 0.001 * d, "c_return": 0.0, "c_squared": 0.0, "c_down_squared": 0.0,
         "base_var": 0.05, "flat_var": 0.048}
        for d in (21, 22, 23)
    ]
    monkeypatch.setattr(D, "load_results", lambda: {"lstm_local": pd.DataFrame(rows)})
    out = A.explain_lstm_local("BTC", 0.025)
    assert [d["date"] for d in out["days"]] == ["2026-09-21", "2026-09-22", "2026-09-23"]
    assert out["asof"] == "2026-09-23" and out["base_var"] == 0.05 and out["flat_var"] == 0.048
    monkeypatch.setattr(D, "load_results", lambda: {"lstm_local": pd.DataFrame()})
    assert A.explain_lstm_local("BTC", 0.025)["days"] == []


def _headroom_bt(exceptions_last_250, aged_in_oldest_30=0, model="M", n=300):
    """A 99% / 97.5% walk-forward with a chosen number of exceptions in the last 250 days."""
    dates = pd.date_range("2025-01-01", periods=n)
    viol = np.zeros(n, bool)
    tail_idx = np.arange(n - 250, n)
    viol[tail_idx[-exceptions_last_250 + aged_in_oldest_30 :][: exceptions_last_250 - aged_in_oldest_30]] = True
    viol[tail_idx[:aged_in_oldest_30]] = True         # these sit in the oldest 30 days of the window
    rows = []
    for alpha, es in ((0.01, -0.08), (0.025, -0.06)):
        rows.append(pd.DataFrame({
            "date": dates, "asset": "BTC", "model": model, "window": 500, "alpha": alpha,
            "var": -0.04, "es": es, "sigma2": 1e-3, "realized": 0.0,
            "violation": viol if alpha == 0.01 else np.zeros(n, bool), "pit": 0.5,
        }))
    return pd.concat(rows, ignore_index=True)


def test_basel_headroom_zone_steps_and_capital(monkeypatch):
    from cryptorisk.api import data as D

    bt = pd.concat([_headroom_bt(4, model="G4"), _headroom_bt(7, aged_in_oldest_30=3, model="A7"),
                    _headroom_bt(10, model="R10")], ignore_index=True)
    monkeypatch.setattr(D, "load_backtests_live", lambda: bt)
    D.basel_headroom.cache_clear()
    by = {r["model"]: r for r in D.basel_headroom("BTC")}
    g, a, r = by["G4"], by["A7"], by["R10"]
    assert (g["exceptions_250d"], g["zone"], g["zone_if_breached"], g["to_next_zone"]) == (4, "green", "amber", 1)
    assert (a["exceptions_250d"], a["zone"], a["to_next_zone"], a["ageing_out_30d"]) == (7, "amber", 3, 3)
    assert (r["zone"], r["to_next_zone"], r["zone_if_breached"]) == ("red", None, "red")
    # one more exception in green: multiplier 1.5 -> 1.9, on 1M * |ES 97.5%| * sqrt(10)
    assert g["m_c"] == 1.5 and abs(g["m_c_if_breached"] - 1.9) < 1e-9
    assert abs(g["extra_capital_if_breached_usd"] - 0.4 * 1_000_000 * 0.06 * 10**0.5) < 1e-6
    D.basel_headroom.cache_clear()


def test_breach_distance_uses_live_spot_and_rejects_offline_models(monkeypatch):
    import pytest
    from fastapi import HTTPException

    from cryptorisk.api import app as A
    from cryptorisk.api import data as D

    monkeypatch.setattr(D, "today_forecast", lambda asset, model, alphas=(): {
        "asof": pd.Timestamp("2026-09-23"), "last_close": 100.0,
        "var_0.01": np.log(0.94), "es_0.01": np.log(0.90), "var_0.025": np.log(0.96), "es_0.025": np.log(0.93)})
    monkeypatch.setattr(D, "live_price", lambda asset: {"price": 98.0})
    out = A.breach_distance("BTC", "GARCH-t", 0.01)
    assert abs(out["var_price"] - 94.0) < 1e-9 and abs(out["dist_var"] - (94 / 98 - 1)) < 1e-9
    monkeypatch.setattr(D, "live_price", lambda asset: None)         # falls back to the last close
    assert abs(A.breach_distance("BTC", "GARCH-t", 0.01)["dist_var"] - (94 / 100 - 1)) < 1e-9
    with pytest.raises(HTTPException) as e:
        A.breach_distance("BTC", "MS-GARCH", 0.01)
    assert e.value.status_code == 404
