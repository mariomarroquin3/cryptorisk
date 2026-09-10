"""End-to-end: every Phase-2a model runs through the engine on real BTC/ETH
returns from the store. Skipped when the store has not been built.
"""

import pandas as pd
import pytest

from cryptorisk.backtest.engine import walk_forward
from cryptorisk.config import load_config, repo_root
from cryptorisk.models.registry import phase2a_models

_DB = repo_root() / load_config()["paths"]["store"]
pytestmark = pytest.mark.skipif(not _DB.exists(), reason="store not built (run `make data`)")


@pytest.fixture(scope="module")
def returns():
    import duckdb

    con = duckdb.connect(str(_DB), read_only=True)
    out = {
        a: con.execute(
            "SELECT date, log_return FROM returns_daily WHERE asset = ? ORDER BY date", [a]
        ).df()
        for a in ("BTC", "ETH")
    }
    con.close()
    return out


@pytest.mark.parametrize("asset", ["BTC", "ETH"])
@pytest.mark.parametrize("model", phase2a_models(), ids=lambda m: m.name)
def test_model_runs_on_real_data(model, asset, returns):
    df = returns[asset]
    res = walk_forward(
        df, model, alphas=[0.025, 0.01], asset=asset, window=500,
        oos_start="2026-05-01",   # ~130 OOS days: a real smoke, still fast
    )
    f = res.frame
    assert len(f) > 100
    assert f["var"].notna().mean() > 0.98
    assert (f.loc[f["var"].notna(), "es"] <= f.loc[f["var"].notna(), "var"] + 1e-9).all()
    for a in (0.025, 0.01):
        assert 0.0 <= res.hit_rate(a) <= 0.30   # loose: just not degenerate
    assert f["date"].min() >= pd.Timestamp("2026-05-01")
