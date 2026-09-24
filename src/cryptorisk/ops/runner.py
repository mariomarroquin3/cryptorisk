"""Detached process that executes one ops job: ``python -m cryptorisk.ops.runner <job_id>``.

Reads the job's steps from ``meta.json``, runs them sequentially with output
appended to ``log.txt``, stops at the first non-zero exit, and records the
outcome. It is deliberately independent of Streamlit so a job outlives the page.
A background thread touches ``heartbeat`` every few seconds so the UI can tell a
busy job from a dead one.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import threading
import time
import traceback

from cryptorisk.config import repo_root
from cryptorisk.ops import jobs


def _update(job_id: str, **fields) -> dict:
    meta = jobs.read_job(job_id) or {}
    meta.update(fields)
    jobs._write_meta(job_id, meta)
    return meta


def _beat(job_id: str, stop: threading.Event) -> None:
    hb = jobs.jobs_dir() / job_id / "heartbeat"
    while not stop.is_set():
        with contextlib.suppress(OSError):
            hb.write_text(str(time.time()), encoding="utf-8")
        stop.wait(3)


def main(job_id: str) -> int:
    meta = json.loads(jobs._meta_path(job_id).read_text(encoding="utf-8"))
    stop = threading.Event()
    threading.Thread(target=_beat, args=(job_id, stop), daemon=True).start()
    _update(job_id, pid=os.getpid())
    log = jobs.jobs_dir() / job_id / "log.txt"
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0  # type: ignore[attr-defined]
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    code = 0
    try:
        with log.open("ab", buffering=0) as out:
            for i, argv in enumerate(meta["steps"]):
                _update(job_id, step=i)
                out.write(f"\n$ {' '.join(argv)}\n".encode())
                proc = subprocess.Popen(  # noqa: S603
                    argv, cwd=repo_root(), stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    env=env, creationflags=flags,
                )
                _update(job_id, child_pid=proc.pid)
                code = proc.wait()
                out.write(f"[exit {code}]\n".encode())
                if code != 0:
                    break
    except Exception:
        traceback.print_exc()  # lands in runner.err
        code = code or 1
    finally:
        stop.set()
    cur = jobs.read_job(job_id) or {}
    if cur.get("status") != "cancelled":  # otherwise the UI already recorded the outcome
        _update(job_id, status="done" if code == 0 else "failed", exit_code=code, finished=time.time())
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
