"""study.run_realized: recompute realized_daily from cached bars_5m, no fetch.

Regression coverage for a real bug: `make realized` shelled out to
`cryptorisk.data.realized`, which has no `__main__` -- the target was a
silent no-op (exit 0, nothing written). This module is the fix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptorisk.data import store


def _make_bars(n_days: int = 5, bars_per_day: int = 288, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    price = 100.0
    start = pd.Timestamp("2020-01-01")
    for d in range(n_days):
        day = start + pd.Timedelta(days=d)
        for b in range(bars_per_day):
            price *= float(np.exp(rng.normal(0, 0.001)))
            rows.append({
                "ts": day + pd.Timedelta(minutes=5 * b),
                "open": price, "high": price, "low": price, "close": price, "volume": 1.0,
            })
    return pd.DataFrame(rows)


def test_run_realized_writes_from_cached_bars(tmp_path, monkeypatch):
    from cryptorisk.study import run_realized

    db = tmp_path / "t.duckdb"
    con = store.connect(str(db))
    store.write_bars_5m(con, "BTC", _make_bars())
    con.close()

    monkeypatch.setattr(run_realized, "load_config", lambda: {
        "assets": ["BTC"], "paths": {"store": str(db)},
        "realized": {"subsample": True, "jump_test": "BNS"},
    })
    monkeypatch.setattr("sys.argv", ["run_realized"])
    run_realized.main()

    con = store.connect(str(db), read_only=True)
    out = con.execute("SELECT * FROM realized_daily WHERE asset = 'BTC' ORDER BY date").df()
    con.close()
    assert len(out) == 5
    assert (out["rv"] > 0).all()
    # the very first bar in the whole series has no preceding bar to diff
    # against, so day 0 has one fewer return than the rest
    assert out["n_bars"].tolist() == [287, 288, 288, 288, 288]


def test_run_realized_skips_an_asset_with_no_cached_bars(tmp_path, monkeypatch, capsys):
    from cryptorisk.study import run_realized

    db = tmp_path / "t.duckdb"
    store.connect(str(db)).close()

    monkeypatch.setattr(run_realized, "load_config", lambda: {
        "assets": ["ETH"], "paths": {"store": str(db)}, "realized": {},
    })
    monkeypatch.setattr("sys.argv", ["run_realized"])
    run_realized.main()
    assert "no cached 5-min bars" in capsys.readouterr().out


def test_run_realized_is_idempotent(tmp_path, monkeypatch):
    """Re-running must replace, not duplicate, each asset's rows."""
    from cryptorisk.study import run_realized

    db = tmp_path / "t.duckdb"
    con = store.connect(str(db))
    store.write_bars_5m(con, "BTC", _make_bars())
    con.close()

    monkeypatch.setattr(run_realized, "load_config", lambda: {
        "assets": ["BTC"], "paths": {"store": str(db)}, "realized": {},
    })
    monkeypatch.setattr("sys.argv", ["run_realized"])
    run_realized.main()
    run_realized.main()

    con = store.connect(str(db), read_only=True)
    n = con.execute("SELECT count(*) FROM realized_daily WHERE asset = 'BTC'").fetchone()[0]
    con.close()
    assert n == 5
