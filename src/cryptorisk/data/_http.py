"""Small HTTP helper: a session with retry/backoff and a courteous UA.

Ingestion modules use :func:`get_json` / :func:`get_bytes`. Kept tiny on
purpose - no extra dependency.
"""

from __future__ import annotations

import time
from typing import Any

import requests

_UA = "cryptorisk/0.1 (research; +https://github.com)"
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": _UA})


def _request(method: str, url: str, *, params=None, timeout: float, tries: int, backoff: float):
    last: Exception | None = None
    for attempt in range(tries):
        try:
            r = _SESSION.request(method, url, params=params, timeout=timeout)
            if r.status_code == 429 or 500 <= r.status_code < 600:
                raise requests.HTTPError(f"{r.status_code} for {r.url}")
            r.raise_for_status()
            return r
        except requests.RequestException as exc:  # noqa: PERF203
            last = exc
            if attempt < tries - 1:
                time.sleep(backoff * (2**attempt))
    raise RuntimeError(f"GET {url} failed after {tries} tries: {last}")


def get_json(url: str, *, params: dict | None = None, timeout: float = 30, tries: int = 4,
             backoff: float = 1.5) -> Any:
    return _request("GET", url, params=params, timeout=timeout, tries=tries, backoff=backoff).json()


def get_bytes(url: str, *, timeout: float = 60, tries: int = 4, backoff: float = 2.0) -> bytes:
    return _request("GET", url, params=None, timeout=timeout, tries=tries, backoff=backoff).content


def url_exists(url: str, *, timeout: float = 20) -> bool:
    try:
        r = _SESSION.head(url, timeout=timeout, allow_redirects=True)
        return r.status_code == 200
    except requests.RequestException:
        return False
