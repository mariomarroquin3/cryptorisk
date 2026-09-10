"""DuckDB store (V2_PLAN §2).

Tables: prices_daily, returns_daily, bars_5m, realized_daily, context_daily,
microstructure_daily. Point-in-time: rows carry ``vintage_ts``; a backtest as
of date ``t`` reads only the vintage available at ``t``.

Phase 1.
"""

from __future__ import annotations

from pathlib import Path

SCHEMA_VERSION = 1


def connect(path: str | Path):
    raise NotImplementedError("store.connect - Phase 1")


def init_schema(con) -> None:
    raise NotImplementedError("store.init_schema - Phase 1")
