"""Automated data-quality checks (V2_PLAN §2).

Calendar gaps, |return| > 40% flags, zero-volume / exchange-outage days,
between-source divergence over a threshold, level jumps between vintages.
Emits a report; the pipeline refuses to proceed on unexplained flags.

Phase 1.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QualityFlag:
    asset: str
    date: str
    kind: str
    detail: str


def check_daily(df) -> list[QualityFlag]:
    raise NotImplementedError("quality.check_daily - Phase 1")
