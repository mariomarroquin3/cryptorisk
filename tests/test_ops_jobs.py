import os
import sys
import time

import pytest

from cryptorisk.ops import jobs


@pytest.fixture(autouse=True)
def _ops_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("CRYPTORISK_OPS_DIR", str(tmp_path))


def _wait(job_id, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        m = jobs.read_job(job_id)
        if m and m["status"] != "running":
            return m
        time.sleep(0.3)
    raise AssertionError("job did not finish")


def _py(code):
    return [sys.executable, "-c", code]


def test_job_runs_steps_in_order_and_logs():
    jid = jobs.start_job("ok", [_py("print('one')"), _py("print('two')")], key="t")
    m = _wait(jid)
    assert m["status"] == "done" and m["exit_code"] == 0
    log = jobs.tail_log(jid)
    assert log.index("one") < log.index("two") and "[exit 0]" in log


def test_job_stops_at_first_failure():
    jid = jobs.start_job("bad", [_py("import sys; sys.exit(3)"), _py("print('never')")], key="t")
    m = _wait(jid)
    assert m["status"] == "failed" and m["exit_code"] == 3
    assert "never" not in jobs.tail_log(jid)


def test_lock_blocks_overlapping_job_then_releases():
    a = jobs.start_job("slow", [_py("import time; time.sleep(4)")], key="a", locks={"store"})
    with pytest.raises(jobs.Busy):
        jobs.start_job("clash", [_py("print(1)")], key="b", locks={"store"})
    jobs.start_job("free", [_py("print(1)")], key="c", locks={"live"})  # different lock is fine
    _wait(a)
    assert "store" not in jobs.running_locks()


def test_cancel_kills_the_job():
    jid = jobs.start_job("long", [_py("import time; time.sleep(60)")], key="t")
    time.sleep(1.5)
    assert jobs.cancel(jid)
    assert jobs.read_job(jid)["status"] == "cancelled"
    time.sleep(1)
    assert not jobs.pid_alive(jobs.read_job(jid)["pid"])


def test_dead_runner_is_marked_lost_only_when_heartbeat_is_stale():
    jid = jobs.start_job("x", [_py("print(1)")], key="t")
    _wait(jid)
    meta = jobs.read_job(jid)
    meta.update(status="running", pid=99999999, started=time.time() - 60)
    jobs._write_meta(jid, meta)
    hb = jobs.jobs_dir() / jid / "heartbeat"
    hb.write_text("x")  # fresh heartbeat: a false-negative pid check must not kill the job
    assert jobs.read_job(jid)["status"] == "running"
    old = time.time() - 120
    os.utime(hb, (old, old))
    assert jobs.read_job(jid)["status"] == "lost"


def test_cancel_also_kills_the_step_process():
    jid = jobs.start_job("long", [_py("import time; time.sleep(60)")], key="t")
    for _ in range(40):
        child = (jobs.read_job(jid) or {}).get("child_pid")
        if child:
            break
        time.sleep(0.25)
    assert child and jobs.pid_alive(child)
    jobs.cancel(jid)
    time.sleep(1.5)
    assert not jobs.pid_alive(child)


def test_catalog_builds_argv_and_options():
    spec, build = jobs.CATALOG["backtests"]
    (argv,) = build({"models": ["HS", "RF-QR"], "assets": ["BTC"]})
    assert argv[-5:] == ["--models", "HS", "RF-QR", "--assets", "BTC"]
    assert spec.frozen and "store" in spec.locks
    (daily,) = jobs.CATALOG["daily"][1]({"no_refresh": True})
    assert daily[-1] == "--no-refresh"


def test_git_calls_open_no_console_window(monkeypatch):
    from cryptorisk.ops import deploy

    seen = []

    def fake_run(argv, **kw):
        seen.append((argv[0], kw.get("creationflags")))

        class P:
            returncode, stdout, stderr = 0, "", ""

        return P()

    monkeypatch.setattr(deploy.subprocess, "run", fake_run)
    deploy._git("status")
    jobs._kill_tree(123456789) if os.name == "nt" else None
    assert seen[0] == ("git", jobs.NO_WINDOW)
    if os.name == "nt":
        assert seen[1] == ("taskkill", jobs.NO_WINDOW) and jobs.NO_WINDOW != 0


def test_git_state_is_cached_until_fetch_or_invalidate(monkeypatch):
    from cryptorisk.ops import deploy

    calls = []
    monkeypatch.setattr(deploy, "_git_state", lambda fetch: calls.append(fetch) or {"n": len(calls)})
    deploy.invalidate_git_state()
    assert deploy.git_state() == {"n": 1} and deploy.git_state() == {"n": 1}
    assert len(calls) == 1
    assert deploy.git_state(fetch=True) == {"n": 2}  # fetch always refreshes
    deploy.invalidate_git_state()
    deploy.git_state()
    assert calls == [False, True, False]
    monkeypatch.setattr(deploy.time, "monotonic", lambda: 1e9)  # long past the TTL
    deploy.git_state()
    assert len(calls) == 4
    deploy.invalidate_git_state()
