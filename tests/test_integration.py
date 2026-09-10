"""End-to-end: every model runs through the engine on real BTC/ETH data from
the store. Skipped when the store has not been built.
"""

import pandas as pd
import pytest

from cryptorisk.backtest.engine import walk_forward
from cryptorisk.config import load_config, repo_root
from cryptorisk.models.registry import all_models

_DB = repo_root() / load_config()["paths"]["store"]
pytestmark = pytest.mark.skipif(not _DB.exists(), reason="store not built (run `make data`)")


@pytest.fixture(scope="module")
def data():
    import duckdb

    con = duckdb.connect(str(_DB), read_only=True)
    out = {}
    for a in ("BTC", "ETH"):
        r = con.execute(
            "SELECT date, log_return FROM returns_daily WHERE asset = ? ORDER BY date", [a]
        ).df()
        rlz = con.execute(
            "SELECT date, rv, bv, rsv_pos, rsv_neg, jump, rq FROM realized_daily "
            "WHERE asset = ? ORDER BY date",
            [a],
        ).df()
        out[a] = r.merge(rlz, on="date", how="left")
    con.close()
    return out


@pytest.mark.parametrize("asset", ["BTC", "ETH"])
@pytest.mark.parametrize("model", all_models(), ids=lambda m: m.name)
def test_model_runs_on_real_data(model, asset, data):
    res = walk_forward(
        data[asset], model, alphas=[0.025, 0.01], asset=asset, window=500,
        oos_start="2026-05-01",   # ~130 OOS days: a real smoke, still fast
    )
    f = res.frame
    assert len(f) > 100
    assert f["var"].notna().mean() > 0.98
    ok = f["var"].notna()
    assert (f.loc[ok, "es"] <= f.loc[ok, "var"] + 1e-9).all()
    assert (f.loc[ok, "var"] < 0).all()
    for a in (0.025, 0.01):
        assert 0.0 <= res.hit_rate(a) <= 0.30
    assert f["date"].min() >= pd.Timestamp("2026-05-01")
