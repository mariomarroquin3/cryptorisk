import numpy as np
import pandas as pd

from cryptorisk.api import live_tail as L


def _bars(start: str, days: int, drift: float = 0.0) -> pd.DataFrame:
    ts = pd.date_range(start, periods=days * 288, freq="5min")
    rng = np.random.default_rng(0)
    px = 100 * np.exp(np.cumsum(rng.normal(drift, 0.001, len(ts))))
    return pd.DataFrame({"ts": ts, "close": px})


def _hist(last: str) -> pd.DataFrame:
    return pd.DataFrame({
        "asset": ["BTC"], "date": [pd.Timestamp(last)], "close": [100.0], "log_return": [0.0],
        "rv": [1e-4], "bv": [1e-4], "rsv_pos": [5e-5], "rsv_neg": [5e-5], "jump": [0.0], "rq": [1e-8],
    })


def test_extend_adds_complete_days_only(monkeypatch):
    bars = _bars("2026-01-01 23:55", 3)          # 23:55 of Jan 1 .. through Jan 4 23:50
    bars = bars[bars["ts"] < "2026-01-04 06:00"]  # Jan 4 incomplete
    monkeypatch.setattr(L, "fetch_5m", lambda *a, **k: bars)
    out = L.extend_history("BTC", _hist("2026-01-01"), today=pd.Timestamp("2026-01-04"))
    assert list(out["date"].dt.day) == [1, 2, 3]
    assert list(out.columns) == list(_hist("2026-01-01").columns)
    assert out["rv"].iloc[1:].notna().all() and (out["rv"].iloc[1:] > 0).all()
    d2 = bars[bars["ts"].dt.normalize() == "2026-01-02"]["close"].iloc[-1]
    assert np.isclose(out["close"].iloc[1], d2)
    assert np.isclose(out["log_return"].iloc[1], np.log(d2 / 100.0))


def test_extend_noop_when_current_or_network_down(monkeypatch):
    h = _hist("2026-01-03")
    assert L.extend_history("BTC", h, today=pd.Timestamp("2026-01-04")) is h
    monkeypatch.setattr(L, "fetch_5m", lambda *a, **k: None)
    assert L.extend_history("BTC", h, today=pd.Timestamp("2026-01-10")) is h


def test_today_so_far(monkeypatch):
    bars = _bars("2026-01-04", 1).iloc[:100]
    monkeypatch.setattr(L, "fetch_5m", lambda *a, **k: bars)
    st = L.today_so_far("BTC", 100.0, now=pd.Timestamp("2026-01-04 08:20"))
    assert st["n_bars"] == 100 and st["low_ret"] <= st["ret_so_far"] <= st["high_ret"]
