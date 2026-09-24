"""Read-only status collectors for the ops center: how fresh is each dataset,
which tools are installed, what did each result file last get built from."""

from __future__ import annotations

import importlib.util
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from cryptorisk.config import load_config, repo_root

#: models whose live re-fit needs torch, so they cannot be refitted on the API host
TORCH_MODELS = {"LSTM-Vol"}
#: models that only exist as an offline R run
R_MODELS = {"MS-GARCH"}


def utc_yesterday() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(UTC).date()) - pd.Timedelta(days=1)


def _behind(last: Any) -> int | None:
    if last is None or pd.isna(last):
        return None
    return int((utc_yesterday() - pd.Timestamp(last).normalize()).days)


def data_status() -> pd.DataFrame:
    """Latest date of every dataset that feeds the deployed app, per asset."""
    cfg = load_config()
    res = repo_root() / cfg["paths"]["results"]
    rows: list[dict[str, Any]] = []

    def add(source: str, asset: str, last: Any, note: str = "") -> None:
        rows.append({"source": source, "asset": asset, "last_date": None if last is None else pd.Timestamp(last).date(),
                     "days_behind": _behind(last), "note": note})

    ph = res / "price_history.parquet"
    if ph.exists():
        df = pd.read_parquet(ph, columns=["asset", "date"])
        for a, g in df.groupby("asset"):
            add("price_history.parquet (API snapshot)", a, g["date"].max())
    bt = res / "backtests.parquet"
    if bt.exists():
        df = pd.read_parquet(bt, columns=["asset", "date"])
        for a, g in df.groupby("asset"):
            add("backtests.parquet (frozen study)", a, g["date"].max(), "frozen on purpose")
    live = res / "backtests_live.parquet"
    if live.exists():
        df = pd.read_parquet(live, columns=["asset", "date"])
        for a, g in df.groupby("asset"):
            add("backtests_live.parquet (live walk-forward)", a, g["date"].max())
    else:
        add("backtests_live.parquet (live walk-forward)", "-", None, "not created yet: run the daily update")

    store = repo_root() / cfg["paths"]["store"]
    if store.exists():
        try:
            import duckdb

            con = duckdb.connect(str(store), read_only=True)
            try:
                for table, col in (("returns_daily", "date"), ("realized_daily", "date"), ("bars_5m", "ts")):
                    q = con.execute(f"SELECT asset, max({col}) FROM {table} GROUP BY asset ORDER BY asset").fetchall()
                    for a, last in q:
                        add(f"store.{table}", a, last, "study sample: frozen" if table != "bars_5m" else "")
            finally:
                con.close()
        except Exception as exc:  # store locked by a running job, or table missing
            add("store (DuckDB)", "-", None, f"unavailable: {type(exc).__name__}")
    return pd.DataFrame(rows)


def results_files() -> pd.DataFrame:
    """When each key result file was last written."""
    cfg = load_config()
    res = repo_root() / cfg["paths"]["results"]
    names = [
        "backtests.parquet", "backtests_live.parquet", "price_history.parquet", "eval_fz0_mcs.csv",
        "eval_coverage.csv", "decision_capital.csv", "decision_estimation_risk.csv", "explain_rf_importance.csv",
        "explain_lstm_importance.csv", "portfolio_eval.csv",
    ]
    rows = []
    for n in names + ["../../docs/results.md"]:
        p = (res / n).resolve()
        if p.exists():
            age = (time.time() - p.stat().st_mtime) / 86400
            rows.append({"file": Path(n).name, "modified": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                         "age_days": round(age, 1)})
        else:
            rows.append({"file": Path(n).name, "modified": "missing", "age_days": None})
    return pd.DataFrame(rows)


def env_status() -> dict[str, str]:
    torch_v = None
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch

            torch_v = torch.__version__
        except Exception as exc:  # broken install
            torch_v = f"error: {type(exc).__name__}"
    return {
        "python": sys.version.split()[0],
        "torch": torch_v or "not installed (LSTM-Vol cannot run)",
        "Rscript": shutil.which("Rscript") or "not found (MS-GARCH cannot run)",
        "node": shutil.which("node") or "not found",
        "git": shutil.which("git") or "not found",
    }


def model_table() -> pd.DataFrame:
    """One row per model: family, where it can be refitted, and how far each
    stored walk-forward reaches."""
    from cryptorisk.models.notes import METHOD_FAMILY
    from cryptorisk.models.registry import all_models

    cfg = load_config()
    res = repo_root() / cfg["paths"]["results"]
    last: dict[str, dict[str, Any]] = {}
    for label, fname in (("frozen_end", "backtests.parquet"), ("live_end", "backtests_live.parquet")):
        p = res / fname
        if p.exists():
            df = pd.read_parquet(p, columns=["model", "date"])
            for m, d in df.groupby("model", observed=True)["date"].max().items():
                last.setdefault(m, {})[label] = pd.Timestamp(d).date()
    have_torch = importlib.util.find_spec("torch") is not None
    rows = []
    for m in all_models():
        name = m.name
        if name in R_MODELS:
            where = "offline R run only"
        elif name in TORCH_MODELS:
            where = "local/CI (torch)" + ("" if have_torch else " - torch missing here")
        else:
            where = "API (live) + local/CI"
        rows.append({
            "model": name, "family": METHOD_FAMILY.get(name, ""), "refit": where,
            "frozen_end": last.get(name, {}).get("frozen_end"), "live_end": last.get(name, {}).get("live_end"),
        })
    df = pd.DataFrame(rows)
    df["live_days_behind"] = [
        _behind(d if pd.notna(d) else f) for d, f in zip(df["live_end"], df["frozen_end"], strict=True)
    ]
    return df


def expected_live_end() -> pd.Timestamp:
    return utc_yesterday()


__all__ = [
    "TORCH_MODELS", "R_MODELS", "data_status", "results_files", "env_status", "model_table",
    "expected_live_end", "utc_yesterday",
]
