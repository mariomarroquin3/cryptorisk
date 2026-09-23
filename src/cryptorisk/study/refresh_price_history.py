"""Roll ``data/results/price_history.parquet`` forward to yesterday (UTC).

Store-free: appends the complete days since the snapshot's last date from
Binance 5-min klines (``api.live_tail``), realized measures included. Run daily
by ``.github/workflows/refresh-data.yml`` so the committed snapshot -- what the
API serves if Binance is unreachable at read time -- never goes stale.

    python -m cryptorisk.study.refresh_price_history
"""

from __future__ import annotations

import pandas as pd

from cryptorisk.api import live_tail
from cryptorisk.config import load_config, repo_root


def refresh() -> pd.DataFrame:
    path = repo_root() / load_config()["paths"]["results"] / "price_history.parquet"
    snap = pd.read_parquet(path)
    snap["date"] = pd.to_datetime(snap["date"])
    parts = [live_tail.extend_history(a, g) for a, g in snap.groupby("asset", sort=False)]
    out = pd.concat(parts, ignore_index=True)
    if len(out) > len(snap):
        out.to_parquet(path, index=False)
    return out


if __name__ == "__main__":
    before = pd.read_parquet(repo_root() / load_config()["paths"]["results"] / "price_history.parquet")
    res = refresh()
    last = res.groupby("asset")["date"].max().dt.date.to_dict()
    print(f"+{len(res) - len(before)} rows; last dates: {last}")
