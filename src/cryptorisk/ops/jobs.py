"""Job runner for the local ops center.

A job is a list of commands (argv lists) run one after another; it stops at the
first failure. Each job runs in a small detached ``cryptorisk.ops.runner``
process, so it keeps going when the Streamlit page reloads or is closed, and its
state lives on disk (``data/ops/jobs/<id>/{meta.json,log.txt}``), not in the UI.

Jobs declare *locks* ("store", "results", "live"); a job that would touch
something a running job is already writing is refused instead of corrupting it.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from cryptorisk.config import repo_root


def jobs_dir() -> Path:
    override = os.environ.get("CRYPTORISK_OPS_DIR")  # tests point this at a temp dir
    d = Path(override) if override else repo_root() / "data" / "ops" / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


#: Windows: run a child without opening a console window (0 elsewhere)
NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0  # type: ignore[attr-defined]


class Busy(RuntimeError):
    """Another running job holds a lock this one needs."""


@dataclass(frozen=True)
class Spec:
    key: str
    label: str
    group: str
    help: str
    est: str = ""
    locks: frozenset[str] = frozenset()
    #: rewrites the frozen study results (changes what the paper/report says)
    frozen: bool = False
    needs: tuple[str, ...] = ()


def _py(*args: str) -> list[str]:
    return [sys.executable, "-u", *args]


def _mod(module: str, *args: str) -> list[str]:
    return _py("-m", module, *args)


def _opt(flag: str, values: Iterable[str] | None) -> list[str]:
    vals = [str(v) for v in (values or [])]
    return [flag, *vals] if vals else []


# ---------------------------------------------------------------------------
# Catalog. `build(params)` returns the argv list for each step.
# ---------------------------------------------------------------------------
CATALOG: dict[str, tuple[Spec, Any]] = {}


def _register(spec: Spec, build) -> None:
    CATALOG[spec.key] = (spec, build)


_register(
    Spec("daily", "Daily update (prices + live walk-forward, all models)", "Live",
         "Rolls price_history forward from Binance 5-min bars, then runs every model's walk-forward for the new days "
         "into backtests_live.parquet, LSTM-Vol included (torch is installed here, not on Render). This is the "
         "'refit' the deployed API cannot do for the LSTM. Afterwards: commit + push from the Deploy tab.",
         est="1-5 min", locks=frozenset({"live"}), needs=("torch",)),
    lambda p: [_mod("cryptorisk.study.extend_backtests", *(["--no-refresh"] if p.get("no_refresh") else []))],
)
_register(
    Spec("prices", "Refresh price history only", "Live",
         "Appends the complete UTC days since the snapshot (with realized measures). No model runs.",
         est="< 1 min", locks=frozenset({"live"})),
    lambda p: [_mod("cryptorisk.study.refresh_price_history")],
)
_register(
    Spec("ingest", "Ingest data (daily + 5-min + context)", "Data",
         "Full store ingest (Phase 1). Slow; downloads monthly Binance archives.",
         est="10-40 min", locks=frozenset({"store"})),
    lambda p: [_mod("cryptorisk.study.run_ingest", *_opt("--assets", p.get("assets")),
                    *(["--end", p["end"]] if p.get("end") else []))],
)
_register(
    Spec("intraday_tail", "Fill in-progress month of 5-min bars", "Data",
         "REST fill for the month data.binance.vision has not published yet, then recompute realized measures.",
         est="1-3 min", locks=frozenset({"store"})),
    lambda p: [_mod("cryptorisk.study.run_ingest", "--intraday-tail", *(["--end", p["end"]] if p.get("end") else []))],
)
_register(
    Spec("realized", "Recompute realized measures", "Data",
         "RV / BV / semivariances / jumps / RQ from the 5-min bars in the store.",
         est="1-5 min", locks=frozenset({"store"})),
    lambda p: [_mod("cryptorisk.study.run_realized", *_opt("--assets", p.get("assets")))],
)
_register(
    Spec("snapshot", "Export price_history from the store", "Data",
         "Rebuilds price_history.parquet from the store. The store ends at the study sample end, so this rolls the "
         "snapshot BACK to that date until the next daily update.",
         est="< 1 min", locks=frozenset({"store", "live"}), frozen=True),
    lambda p: [_mod("cryptorisk.study.export_price_history")],
)
_register(
    Spec("backtests", "Walk-forward backtests (frozen study)", "Study",
         "Re-runs the study's walk-forward for the chosen models into backtests.parquet. This is the FROZEN sample: "
         "use it after changing a model, and re-run evaluate/decide/report afterwards.",
         est="minutes to hours", locks=frozenset({"store", "results"}), frozen=True),
    lambda p: [_mod("cryptorisk.study.run_backtests", *_opt("--models", p.get("models")),
                    *_opt("--assets", p.get("assets")))],
)
_register(
    Spec("msgarch", "MS-GARCH walk-forward (R)", "Study",
         "The R MSGARCH walk-forward (~8 min per asset window); its predictions feed the MS-GARCH model.",
         est="10-60 min", locks=frozenset({"store", "results"}), frozen=True, needs=("R",)),
    lambda p: [_mod("cryptorisk.study.run_msgarch")],
)
_register(
    Spec("evaluate", "Evaluation battery", "Study",
         "Coverage tests, ES tests, FZ0 + Model Confidence Set, GW-CPA, PIT.",
         est="2-10 min", locks=frozenset({"results"}), frozen=True),
    lambda p: [_mod("cryptorisk.study.run_evaluation", *_opt("--assets", p.get("assets")))],
)
_register(
    Spec("subperiods", "Sub-period re-evaluation", "Study", "COVID / Luna / FTX / calm windows + CPA.",
         est="1-5 min", locks=frozenset({"results"}), frozen=True),
    lambda p: [_mod("cryptorisk.study.subperiods")],
)
_register(
    Spec("decide", "Decision layer", "Study",
         "Capital (FRTB ES), limits, PLA, hedge, estimation-risk bands (RF-QR and LSTM-Vol included).",
         est="5-15 min", locks=frozenset({"results", "store"}), frozen=True, needs=("torch",)),
    lambda p: [_mod("cryptorisk.study.run_decision")],
)
_register(
    Spec("portfolio", "Portfolio extension", "Study", "BTC+ETH basket VaR/ES with copula tails.",
         est="5-20 min", locks=frozenset({"results", "store"}), frozen=True),
    lambda p: [_mod("cryptorisk.study.run_portfolio")],
)
_register(
    Spec("explain", "Explainability (RF importance + LSTM permutation)", "Study",
         "RF-QR importance/ESS diagnostics and LSTM-Vol permutation importance. Retrains the LSTM every N windows.",
         est="5-30 min", locks=frozenset({"results", "store"}), needs=("torch",)),
    lambda p: [_mod("cryptorisk.study.run_explain", "--lstm-every", str(int(p.get("lstm_every", 60))),
                    *(["--skip-lstm"] if p.get("skip_lstm") else []), *(["--skip-rf"] if p.get("skip_rf") else []))],
)
_register(
    Spec("report", "Assemble report", "Study", "docs/results.md, model cards and figures from data/results.",
         est="< 1 min", locks=frozenset({"results"}), frozen=True),
    lambda p: [_mod("cryptorisk.study.report")],
)
_register(
    Spec("selftest", "Runner self-test (25 s)", "Dev",
         "Checks that a job survives page reloads and is tracked to completion. Does nothing else.", est="25 s"),
    lambda p: [_py("-c", "import time; from joblib import Parallel, delayed; print('working'); "
                         "print(Parallel(n_jobs=-1)(delayed(time.sleep)(2) for _ in range(8))); "
                         "time.sleep(20); print('done')")],
)
_register(
    Spec("tests", "Run test suite (ruff + pytest)", "Dev", "Lint and the full test suite.",
         est="~10 min"),
    lambda p: [_py("-m", "ruff", "check", "."), _py("-m", "pytest", "-q")],
)


def catalog() -> list[Spec]:
    return [s for s, _ in CATALOG.values()]


# ---------------------------------------------------------------------------
# Process helpers
# ---------------------------------------------------------------------------
def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    if os.name == "nt":
        import ctypes

        k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        h = k32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(k32.GetExitCodeProcess(h, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    # A finished child nobody has reaped yet is a zombie: it still answers signal 0 but is not running.
    try:
        state = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return True  # no /proc (macOS): trust the signal check
    return state != "Z"


def _kill_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(  # noqa: S603
            ["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False,
            creationflags=NO_WINDOW,  # otherwise every call flashes a console window
        )
    else:
        import signal

        with contextlib.suppress(OSError):
            os.killpg(os.getpgid(pid), signal.SIGTERM)


# ---------------------------------------------------------------------------
# Job lifecycle
# ---------------------------------------------------------------------------
def _meta_path(job_id: str) -> Path:
    return jobs_dir() / job_id / "meta.json"


def _write_meta(job_id: str, meta: dict[str, Any]) -> None:
    p = _meta_path(job_id)
    tmp = p.with_name(f"meta.{os.getpid()}.tmp")  # per-process temp: two writers never share one
    tmp.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    for attempt in range(8):  # on Windows the target can be briefly held open by a reader
        try:
            os.replace(tmp, p)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.05 * (attempt + 1))


HEARTBEAT_STALE_S = 45


def _heartbeat_age(job_id: str) -> float | None:
    hb = jobs_dir() / job_id / "heartbeat"
    return time.time() - hb.stat().st_mtime if hb.exists() else None


def read_job(job_id: str) -> dict[str, Any] | None:
    p = _meta_path(job_id)
    if not p.exists():
        return None
    try:
        meta = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    # a running job whose runner died without recording an outcome (killed, crashed, machine slept)
    hb = _heartbeat_age(job_id)
    stale = hb is None or hb > HEARTBEAT_STALE_S
    if (
        meta.get("status") == "running"
        and stale
        and not pid_alive(meta.get("pid"))
        and time.time() - meta.get("started", 0) > 5
    ):
        meta.update(status="lost", finished=time.time())
        _write_meta(job_id, meta)
    return meta


def list_jobs(limit: int = 30) -> list[dict[str, Any]]:
    ids = sorted((d.name for d in jobs_dir().iterdir() if d.is_dir()), reverse=True)[:limit]
    return [m for i in ids if (m := read_job(i)) is not None]


def running_locks() -> dict[str, str]:
    """lock name -> id of the running job holding it."""
    held: dict[str, str] = {}
    for m in list_jobs(50):
        if m.get("status") == "running":
            for lock in m.get("locks", []):
                held[lock] = m["id"]
    return held


def start_job(
    label: str,
    steps: list[list[str]],
    *,
    key: str = "custom",
    locks: Iterable[str] = (),
    params: dict[str, Any] | None = None,
) -> str:
    locks = sorted(set(locks))
    held = running_locks()
    clash = [lk for lk in locks if lk in held]
    if clash:
        raise Busy(f"{', '.join(clash)} in use by job {held[clash[0]]}")
    # the short random suffix keeps two jobs of the same kind started in the same second apart
    job_id = f"{datetime.now():%Y%m%d-%H%M%S}-{key}-{uuid.uuid4().hex[:4]}"
    (jobs_dir() / job_id).mkdir(parents=True, exist_ok=False)
    meta = {
        "id": job_id, "key": key, "label": label, "steps": steps, "locks": locks,
        "params": params or {}, "status": "running", "started": time.time(),
        "step": 0, "exit_code": None, "pid": None,
    }
    _write_meta(job_id, meta)
    flags = 0
    if os.name == "nt":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    with (jobs_dir() / job_id / "runner.err").open("wb") as err:  # a crashing runner leaves its traceback here
        proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "cryptorisk.ops.runner", job_id],
            cwd=repo_root(), creationflags=flags, start_new_session=os.name != "nt",
            stdin=subprocess.DEVNULL, stdout=err, stderr=err, close_fds=True,
        )
    meta["pid"] = proc.pid
    cur = read_job(job_id) or meta
    if cur.get("pid") is None:  # the runner records its own pid too; don't clobber a finished job
        cur["pid"] = proc.pid
        _write_meta(job_id, cur)
    return job_id


def start(key: str, params: dict[str, Any] | None = None) -> str:
    spec, build = CATALOG[key]
    params = params or {}
    return start_job(spec.label, build(params), key=key, locks=spec.locks, params=params)


def cancel(job_id: str) -> bool:
    meta = read_job(job_id)
    if not meta or meta.get("status") != "running":
        return False
    for key in ("child_pid", "pid"):  # the step first, then the runner (which may already be gone)
        if meta.get(key):
            _kill_tree(int(meta[key]))
    meta.update(status="cancelled", finished=time.time())
    _write_meta(job_id, meta)
    return True


def runner_errors(job_id: str) -> str:
    p = jobs_dir() / job_id / "runner.err"
    return p.read_text(encoding="utf-8", errors="replace").strip() if p.exists() else ""


def tail_log(job_id: str, n: int = 200) -> str:
    p = jobs_dir() / job_id / "log.txt"
    if not p.exists():
        return ""
    with p.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(0, size - 200_000))
        data = fh.read().decode("utf-8", errors="replace")
    return "\n".join(data.splitlines()[-n:])
