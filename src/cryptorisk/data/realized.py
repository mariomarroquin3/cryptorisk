"""Realized measures from 5-minute bars (V2_PLAN §2, §3).

realized variance (with subsampling), bipower variation, realized
semivariance (+/-), and the jump component via Barndorff-Nielsen & Shephard
(2006) or Lee & Mykland (2008). High-frequency cleaning follows
Barndorff-Nielsen, Hansen, Lunde & Shephard (2009).

Phase 1.
"""

from __future__ import annotations

import pandas as pd


def realized_daily(bars_5m: pd.DataFrame, *, subsample: bool = True, jump_test: str = "BNS") -> pd.DataFrame:
    raise NotImplementedError("realized.realized_daily - Phase 1")


if __name__ == "__main__":  # `make realized`
    raise SystemExit(realized_daily.__doc__)
