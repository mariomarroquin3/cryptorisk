"""A minimal in-process TTL cache -- the API's equivalent of the dashboard's
``st.cache_data``/``st.cache_resource``, without a Streamlit dependency.

Not for multi-process deployments (state is per-worker); fine for a single
``uvicorn`` process serving a local/personal tool.

Two guarantees beyond a plain dict-with-expiry:

* **Single flight.** Concurrent callers that miss the same key wait for the first
  computation instead of each repeating it (a cold start used to fetch the price
  history once per simultaneous request).
* **Stale-while-revalidate** (opt in): once a value exists, an expired hit is
  returned immediately and refreshed in a background thread, so the request that
  happens to land right after the TTL no longer pays the full recompute.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def ttl_cache(seconds: float, *, stale_while_revalidate: bool = False) -> Callable[[F], F]:
    def decorator(fn: F) -> F:
        store: dict[tuple, tuple[float, Any]] = {}
        locks: dict[tuple, threading.Lock] = {}
        guard = threading.Lock()

        def lock_for(key: tuple) -> threading.Lock:
            with guard:
                return locks.setdefault(key, threading.Lock())

        def refresh(key: tuple, args: tuple, kwargs: dict) -> None:
            lock = lock_for(key)
            if not lock.acquire(blocking=False):  # a refresh is already running
                return
            try:
                store[key] = (time.monotonic(), fn(*args, **kwargs))
            except Exception:  # noqa: BLE001 - keep serving the stale value; the next call retries
                pass
            finally:
                lock.release()

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = (args, tuple(sorted(kwargs.items())))
            hit = store.get(key)
            if hit is not None and time.monotonic() - hit[0] < seconds:
                return hit[1]
            if hit is not None and stale_while_revalidate:
                threading.Thread(target=refresh, args=(key, args, kwargs), daemon=True).start()
                return hit[1]
            with lock_for(key):
                hit = store.get(key)  # another caller may have filled it while we waited
                if hit is not None and time.monotonic() - hit[0] < seconds:
                    return hit[1]
                value = fn(*args, **kwargs)
                store[key] = (time.monotonic(), value)
                return value

        wrapper.cache_clear = store.clear  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator
