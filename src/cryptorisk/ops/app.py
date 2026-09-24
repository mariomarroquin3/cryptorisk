"""Cryptorisk Ops Center -- a localhost control panel for the pipeline.

    make ops          # http://127.0.0.1:8502

Runs on your machine only (bound to 127.0.0.1). It launches the same commands
as the Makefile, tracks them as jobs that survive page reloads, and shows what is
stale, what is running and what is deployed. Nothing is committed or pushed
without an explicit confirmation.
"""

from __future__ import annotations

import time
from datetime import datetime

import pandas as pd
import streamlit as st

from cryptorisk.config import load_config
from cryptorisk.dashboard.theme import page_setup
from cryptorisk.ops import deploy, jobs, status

page_setup("Ops Center")

_ICON = {"running": "🟡", "done": "🟢", "failed": "🔴", "cancelled": "⚪", "lost": "🟠"}


def _fmt_dur(m: dict) -> str:
    end = m.get("finished") or time.time()
    s = int(end - m.get("started", end))
    return f"{s // 60}m {s % 60:02d}s"


def _started(m: dict) -> str:
    return datetime.fromtimestamp(m["started"]).strftime("%m-%d %H:%M:%S")


def _start(key: str, params: dict | None = None) -> None:
    try:
        jid = jobs.start(key, params)
    except jobs.Busy as exc:
        st.error(f"Cannot start: {exc}")
        return
    st.session_state["watch"] = jid
    st.success(f"Started {jid}. Follow it in the Jobs tab.")


def _color_behind(v):
    if v is None or pd.isna(v):
        return ""
    return "color: #3ddc84" if v <= 1 else ("color: #ffb020" if v <= 3 else "color: #ff4d4f")


st.title("◈ Cryptorisk Ops Center")
st.caption("Local control panel. Stops when you close the terminal; the jobs it launches keep running.")

env = status.env_status()
with st.sidebar:
    st.subheader("Environment")
    for k, v in env.items():
        st.write(f"**{k}**: {v}")
    running = [m for m in jobs.list_jobs(50) if m.get("status") == "running"]
    st.subheader("Running jobs")
    st.write("\n".join(f"- {m['label']}" for m in running) if running else "none")
    if st.button("Refresh"):
        st.rerun()

tab_status, tab_daily, tab_pipe, tab_models, tab_jobs, tab_deploy = st.tabs(
    ["Status", "Daily update", "Pipeline", "Models", "Jobs", "Deploy"]
)

# --------------------------------------------------------------------------- Status
with tab_status:
    ds = status.data_status()
    live = ds[ds["source"].str.startswith(("price_history", "backtests_live"))]
    worst = live["days_behind"].max() if not live.empty and live["days_behind"].notna().any() else None
    c = st.columns(4)
    c[0].metric("Expected last day (UTC)", str(status.utc_yesterday().date()))
    c[1].metric("Live data behind by", "n/a" if worst is None else f"{int(worst)} d")
    gs = deploy.git_state()
    c[2].metric("Unpushed commits", "n/a" if gs["ahead"] is None else gs["ahead"])
    c[3].metric("Uncommitted files", len(gs["files"]))
    if worst is not None and worst > 1:
        st.warning("Live data is stale. Run the daily update, then commit and push from the Deploy tab.")
    st.subheader("Datasets")
    styled = ds.style.map(_color_behind, subset=["days_behind"])
    st.dataframe(styled, hide_index=True, width="stretch")
    st.caption("The frozen study and the store stay at the study sample end on purpose (2026-09-10).")
    st.subheader("Result files")
    st.dataframe(status.results_files(), hide_index=True, width="stretch")

# --------------------------------------------------------------------------- Daily
with tab_daily:
    spec, _ = jobs.CATALOG["daily"]
    st.subheader("Daily update")
    st.write(spec.help)
    st.markdown(
        "**What 'refit' means here.** Every model is re-fitted each day inside the walk-forward. The deployed API "
        "re-fits all of them live *except* LSTM-Vol (no torch on Render) and MS-GARCH (offline R). This job runs the "
        "walk-forward locally for all of them and writes `backtests_live.parquet`, which the API falls back to. "
        "The GitHub workflow *Refresh data* does the same automatically every day once Actions has write permission; "
        "use this button when it has not run or to force a refresh."
    )
    no_refresh = st.checkbox("Skip rolling prices forward (use the current price_history)", value=False)
    missing = [n for n in spec.needs if n == "torch" and "not installed" in env["torch"]]
    if missing:
        st.error("torch is not installed in this environment: LSTM-Vol would be skipped.")
    if st.button("Run daily update", type="primary"):
        _start("daily", {"no_refresh": no_refresh})
    st.markdown("**After it finishes:** open *Deploy* → commit the changed parquet files → push → verify production.")

# --------------------------------------------------------------------------- Pipeline
with tab_pipe:
    cfg = load_config()
    assets = list(cfg["assets"])
    model_names = [m for m in status.model_table()["model"]]
    held = jobs.running_locks()
    st.caption("Same stages as the Makefile. Stages marked FROZEN rewrite the study results: the paper's numbers change.")
    for group in ("Live", "Data", "Study", "Dev"):
        specs = [s for s in jobs.catalog() if s.group == group and s.key != "daily"]
        if not specs:
            continue
        st.subheader(group)
        for s in specs:
            title = f"{s.label}  ·  {s.est}" + ("  ·  ⚠ FROZEN" if s.frozen else "")
            with st.expander(title):
                st.write(s.help)
                p: dict = {}
                if s.key in ("backtests", "ingest", "realized", "evaluate"):
                    p["assets"] = st.multiselect("Assets", assets, default=assets, key=f"a_{s.key}")
                if s.key == "backtests":
                    p["models"] = st.multiselect("Models (empty = all)", model_names, key="m_backtests")
                if s.key in ("ingest", "intraday_tail"):
                    e = st.text_input("End date (YYYY-MM-DD, empty = default)", key=f"e_{s.key}")
                    p["end"] = e.strip() or None
                if s.key == "explain":
                    p["lstm_every"] = st.number_input("Retrain LSTM every N windows", 10, 500, 60, key="le")
                    p["skip_lstm"] = st.checkbox("Skip LSTM", key="sl")
                    p["skip_rf"] = st.checkbox("Skip RF", key="sr")
                clash = [lk for lk in s.locks if lk in held]
                ok = True
                if s.frozen:
                    ok = st.checkbox("I understand this rewrites the frozen study results", key=f"c_{s.key}")
                if "R" in s.needs and "not found" in env["Rscript"]:
                    st.error("Rscript not found.")
                    ok = False
                if "torch" in s.needs and "not installed" in env["torch"]:
                    st.error("torch not installed.")
                    ok = False
                if clash:
                    st.warning(f"Busy: {', '.join(clash)} held by {held[clash[0]]}")
                if st.button(f"Run: {s.label}", key=f"run_{s.key}", disabled=not ok or bool(clash)):
                    _start(s.key, p)

# --------------------------------------------------------------------------- Models
with tab_models:
    mt = status.model_table()
    st.subheader("Models")
    st.dataframe(mt.style.map(_color_behind, subset=["live_days_behind"]), hide_index=True, width="stretch")
    st.caption(
        "refit = where the model can be re-fitted: 'API (live)' models are re-fitted by the deployed API on every "
        "request; the others depend on the daily update. frozen_end / live_end = last day of each stored walk-forward."
    )
    st.subheader("Re-run the frozen backtest for a model")
    st.write(
        "Use after changing a model's code. This rewrites that model's rows in `backtests.parquet` (the frozen "
        "study); re-run *Evaluation → Decision → Report* afterwards."
    )
    sel = st.multiselect("Models", list(mt["model"]), key="m_rerun")
    ok = st.checkbox("I understand this rewrites the frozen study results", key="c_rerun")
    if st.button("Re-run frozen backtest for selected", disabled=not (sel and ok)):
        _start("backtests", {"models": sel})

# --------------------------------------------------------------------------- Jobs
with tab_jobs:
    hist = jobs.list_jobs(30)
    if not hist:
        st.info("No jobs yet.")
    else:
        ids = [m["id"] for m in hist]
        default = ids.index(st.session_state["watch"]) if st.session_state.get("watch") in ids else 0
        pick = st.selectbox(
            "Job", ids, index=default,
            format_func=lambda i: next(
                f"{_ICON.get(m['status'], '?')} {_started(m)}  {m['label']}  ({m['status']})" for m in hist if m["id"] == i
            ),
        )

        @st.fragment(run_every="2s")
        def monitor(job_id: str) -> None:
            m = jobs.read_job(job_id)
            if not m:
                return
            c = st.columns(4)
            c[0].metric("Status", f"{_ICON.get(m['status'], '')} {m['status']}")
            c[1].metric("Elapsed", _fmt_dur(m))
            c[2].metric("Step", f"{m.get('step', 0) + 1}/{len(m['steps'])}")
            c[3].metric("Exit code", "-" if m.get("exit_code") is None else m["exit_code"])
            if m["status"] == "running" and st.button("Cancel job", key=f"cancel_{job_id}"):
                jobs.cancel(job_id)
                st.rerun()
            st.code(jobs.tail_log(job_id, 200) or "(no output yet)", language="text")
            if err := jobs.runner_errors(job_id):
                st.error("The job runner reported an error:")
                st.code(err, language="text")

        monitor(pick)

# --------------------------------------------------------------------------- Deploy
with tab_deploy:
    st.subheader("Git")
    gs = deploy.git_state(fetch=st.button("Fetch remote state"))
    st.write(f"Branch **{gs['branch']}** · ahead {gs['ahead']} · behind {gs['behind']} · last: `{gs['last']}`")
    if gs["unpushed"]:
        st.write("Unpushed:")
        st.code("\n".join(gs["unpushed"]), language="text")
    files = [f for f in gs["files"] if not f["path"].startswith(".claude")]
    if files:
        st.write("Changed files:")
        chosen = st.multiselect(
            "Include in the commit", [f["path"] for f in files], default=[f["path"] for f in files if f["path"].startswith("data/results")],
        )
        msg = st.text_input("Commit message", value="data: roll price history and live walk-forward forward")
        st.caption("A Co-Authored-By trailer is stripped from any message.")
        if st.button("Commit selected files", disabled=not chosen):
            ok, out = deploy.commit(chosen, msg)
            (st.success if ok else st.error)(out or "committed")
    else:
        st.info("Working tree clean.")

    st.divider()
    st.subheader("Push")
    st.write("Pushing to `main` triggers the Render (API) and Vercel (web) deploys.")
    confirm = st.text_input('Type "push" to enable the button')
    if st.button("Push to origin", disabled=confirm.strip().lower() != "push" or not gs["ahead"]):
        ok, out = deploy.push()
        (st.success if ok else st.error)(out or "pushed")

    st.divider()
    st.subheader("Verify production")
    st.caption("Render may take 30-60 s to wake up and a couple of minutes to redeploy after a push.")
    if st.button("Run production checks"):
        with st.spinner("Checking..."):
            res = deploy.verify_production()
        for r in res:
            (st.success if r["ok"] else st.error)(f"**{r['check']}** — {r['detail']}")
