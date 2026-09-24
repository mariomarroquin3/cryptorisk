"""Git and production checks for the ops center.

Nothing here runs on its own: commit and push are explicit calls the UI makes
after the user confirms. Commit messages never carry a Co-Authored-By trailer.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from typing import Any

import requests

from cryptorisk.config import repo_root
from cryptorisk.ops import jobs
from cryptorisk.ops.status import utc_yesterday

API_URL = os.environ.get("CRYPTORISK_API_URL", "https://cryptorisk-api.onrender.com").rstrip("/")
WEB_URL = os.environ.get("CRYPTORISK_WEB_URL", "https://cryptorisk-mauve.vercel.app").rstrip("/")


def _git(*args: str, timeout: float = 60) -> tuple[int, str]:
    p = subprocess.run(  # noqa: S603
        ["git", *args], cwd=repo_root(), capture_output=True, text=True, timeout=timeout, check=False,
        creationflags=jobs.NO_WINDOW,  # a console window per git call piled up on every page reload
    )
    # rstrip only newlines: `git status --porcelain` lines start with a significant space
    return p.returncode, (p.stdout + p.stderr).rstrip("\r\n")


_STATE_TTL_S = 8.0
_FETCH_EVERY_S = 300.0  # refresh the remote refs now and then so "behind" is not stale
# `fetched` starts at -inf, not 0: time.monotonic() counts from boot, so on a freshly started
# machine "now - 0" can be under the interval and the first automatic fetch would be skipped.
_state_cache: dict[str, Any] = {"at": 0.0, "value": None, "fetched": float("-inf")}
_state_lock = threading.Lock()


def invalidate_git_state() -> None:
    with _state_lock:
        _state_cache["value"] = None


def git_state(fetch: bool = False) -> dict[str, Any]:
    """Repo state, cached for a few seconds: Streamlit re-runs the whole page on
    every interaction and each state costs half a dozen git processes. ``fetch``
    always refreshes."""
    with _state_lock:
        hit = _state_cache["value"]
        if not fetch and hit is not None and time.monotonic() - _state_cache["at"] < _STATE_TTL_S:
            return hit
    fetch = fetch or time.monotonic() - _state_cache["fetched"] > _FETCH_EVERY_S
    value = _git_state(fetch)
    with _state_lock:
        _state_cache.update(at=time.monotonic(), value=value)
        if fetch:
            _state_cache["fetched"] = time.monotonic()
    return value


def _git_state(fetch: bool) -> dict[str, Any]:
    if fetch:
        _git("fetch", "--quiet", timeout=120)
    _, branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    branch = branch.strip()
    _, porcelain = _git("status", "--porcelain")
    files = []
    for line in porcelain.splitlines():
        if len(line) > 3:
            files.append({"status": line[:2].strip() or "?", "path": line[3:].strip().strip('"')})
    rc, counts = _git("rev-list", "--left-right", "--count", "@{u}...HEAD")
    behind = ahead = None
    if rc == 0 and counts.split():
        behind, ahead = (int(x) for x in counts.split()[:2])
    _, last = _git("log", "-1", "--format=%h %s")
    _, unpushed = _git("log", "@{u}..HEAD", "--format=%h %s") if ahead else (0, "")
    return {"branch": branch, "files": files, "behind": behind, "ahead": ahead, "last": last,
            "unpushed": [ln for ln in unpushed.splitlines() if ln]}


def clean_message(msg: str) -> str:
    """Drop any Co-Authored-By trailer: this repo's commits carry no attribution."""
    lines = [ln for ln in msg.splitlines() if not ln.lower().startswith("co-authored-by")]
    return "\n".join(lines).strip()


def commit(paths: list[str], message: str) -> tuple[bool, str]:
    msg = clean_message(message)
    if not msg:
        return False, "empty commit message"
    if not paths:
        return False, "no files selected"
    rc, out = _git("add", "--", *paths)
    if rc != 0:
        return False, out
    rc, out = _git("commit", "-m", msg, "--", *paths)
    invalidate_git_state()
    return rc == 0, out


def pull() -> tuple[bool, str]:
    """Fast-forward only: the daily bot pushes data commits, and a merge of its binary
    parquet files would only produce conflicts."""
    rc, out = _git("pull", "--ff-only", timeout=180)
    invalidate_git_state()
    return rc == 0, out


def push() -> tuple[bool, str]:
    rc, out = _git("push", timeout=180)
    invalidate_git_state()
    return rc == 0, out


def _get(path: str, base: str = API_URL, timeout: float = 90) -> requests.Response:
    return requests.get(base + path, timeout=timeout)


def verify_production() -> list[dict[str, Any]]:
    """Checks the deployed app is up AND current. Each row: name, ok, detail."""
    want = utc_yesterday().date()
    out: list[dict[str, Any]] = []

    def check(name: str, fn) -> None:
        try:
            ok, detail = fn()
        except Exception as exc:
            ok, detail = False, f"{type(exc).__name__}: {exc}"[:160]
        out.append({"check": name, "ok": bool(ok), "detail": detail})

    check("API health", lambda: (lambda r: (r.status_code == 200, f"HTTP {r.status_code}"))(_get("/health")))
    check("live price", lambda: (lambda r: (r.status_code == 200, f"HTTP {r.status_code}"))(_get("/price/BTC")))

    def forecast():
        r = _get("/forecast/BTC?alpha=0.025")
        d = r.json()
        asof = str(d.get("asof", d.get("date", "")))[:10]
        return asof >= str(want), f"{d.get('model')} asof {asof} (want >= {want}), source {d.get('source')}"

    check("forecast is current", forecast)

    def live_bt():
        r = _get("/backtests?asset=BTC&model=HS&alpha=0.025&limit=1&live=true")
        last = str(r.json()[-1]["date"])[:10] if r.json() else ""
        return last >= str(want), f"live walk-forward ends {last} (want >= {want})"

    check("live walk-forward is current", live_bt)

    def lstm():
        r = _get("/models/latest?asset=BTC&alpha=0.025")
        row = next((x for x in r.json() if x["model"] == "LSTM-Vol"), None)
        last = str(row["date"])[:10] if row else ""
        return last >= str(want), f"LSTM-Vol latest row {last or 'missing'}"

    check("LSTM-Vol latest row", lstm)
    check("track record", lambda: (lambda d: (len(d.get("days", [])) > 0, f"{len(d.get('days', []))} live days"))(
        _get("/live/track-record?asset=BTC&alpha=0.025").json()))
    check("web (Vercel)", lambda: (lambda r: (r.status_code == 200, f"HTTP {r.status_code}"))(_get("/", WEB_URL, 30)))
    return out
