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
