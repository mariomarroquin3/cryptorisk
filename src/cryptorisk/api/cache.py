"""A minimal in-process TTL cache -- the API's equivalent of the dashboard's
``st.cache_data``/``st.cache_resource``, without a Streamlit dependency.

Not for multi-process deployments (state is per-worker); fine for a single
``uvicorn`` process serving a local/personal tool.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def ttl_cache(seconds: float) -> Callable[[F], F]:
    def decorator(fn: F) -> F:
        store: dict[tuple, tuple[float, Any]] = {}

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = (args, tuple(sorted(kwargs.items())))
            now = time.monotonic()
            hit = store.get(key)
            if hit is not None and now - hit[0] < seconds:
                return hit[1]
            value = fn(*args, **kwargs)
            store[key] = (now, value)
            return value

        wrapper.cache_clear = store.clear  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator
