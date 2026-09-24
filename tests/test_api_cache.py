import threading
import time

from cryptorisk.api.cache import ttl_cache


def test_single_flight_concurrent_misses_compute_once():
    calls = []

    @ttl_cache(60)
    def slow(x):
        calls.append(x)
        time.sleep(0.3)
        return x * 2

    out = []
    threads = [threading.Thread(target=lambda: out.append(slow(21))) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert out == [42] * 6 and calls == [21]


def test_stale_while_revalidate_returns_old_value_then_refreshes():
    n = {"v": 0}

    @ttl_cache(0.05, stale_while_revalidate=True)
    def val():
        n["v"] += 1
        time.sleep(0.2)
        return n["v"]

    assert val() == 1                       # first call is synchronous
    time.sleep(0.1)                         # expired
    t0 = time.monotonic()
    assert val() == 1                       # stale value, immediately
    assert time.monotonic() - t0 < 0.1
    time.sleep(0.5)                         # background refresh finished
    assert val() == 2


def test_failed_refresh_keeps_serving_the_stale_value():
    state = {"boom": False}

    @ttl_cache(0.05, stale_while_revalidate=True)
    def val():
        if state["boom"]:
            raise RuntimeError("network down")
        return "ok"

    assert val() == "ok"
    state["boom"] = True
    time.sleep(0.1)
    assert val() == "ok"
    time.sleep(0.1)
    assert val() == "ok"


def test_plain_cache_still_expires_and_clears():
    n = {"v": 0}

    @ttl_cache(0.05)
    def val():
        n["v"] += 1
        return n["v"]

    assert val() == 1 and val() == 1
    time.sleep(0.1)
    assert val() == 2
    val.cache_clear()
    assert val() == 3


def test_whatif_refits_are_concurrency_bounded(monkeypatch):
    import numpy as np
    import pandas as pd

    from cryptorisk.api import data as D

    n = 560
    r = np.random.default_rng(3).standard_t(6, n) * 0.02
    rv = r**2 + 1e-5
    win = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n), "close": 100 * np.exp(np.cumsum(r)),
        "log_return": r, "rv": rv, "bv": rv * 0.9, "rsv_pos": rv / 2, "rsv_neg": rv / 2, "jump": 0.0, "rq": rv**2,
    })
    monkeypatch.setattr(D, "load_price_window", lambda asset, n=900: win.tail(n))
    live = {"now": 0, "peak": 0}
    lock = threading.Lock()

    class Slow:
        name = "Slow"

        def fit_predict(self, ctx):
            with lock:
                live["now"] += 1
                live["peak"] = max(live["peak"], live["now"])
            time.sleep(0.15)
            with lock:
                live["now"] -= 1
            raise RuntimeError("stop here: only concurrency matters")

    monkeypatch.setitem(D._model_map(), "Slow", Slow())
    D.whatif_forecast.cache_clear()
    threads = [threading.Thread(target=D.whatif_forecast, args=("BTC", "Slow", s / 100)) for s in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert live["peak"] <= 2
    D._model_map().pop("Slow", None)
    D.whatif_forecast.cache_clear()
