"""MSGarchBridge: cache lookup and graceful fallback."""

import numpy as np
import pandas as pd
import pytest

from cryptorisk.models.base import Context, EmpiricalDist, QuantileDist
from cryptorisk.models.msgarch_bridge import MSGarchBridge


def _ctx(asof, asset="BTC"):
    r = np.random.default_rng(0).standard_t(6, 400) * 0.02
    dates = pd.date_range("2019-01-01", periods=400).to_numpy()
    return Context(returns=r, dates=dates, asof=np.datetime64(pd.Timestamp(asof)), asset=asset)


def test_bridge_serves_cached_row():
    cache = pd.DataFrame([{
        "asset": "BTC", "prev_date": "2020-03-11", "date": "2020-03-12",
        "var_0025": -0.09, "es_0025": -0.12, "var_001": -0.14, "es_001": -0.17,
        "sigma2": 0.0016,
    }])
    b = MSGarchBridge(cache)
    d = b.fit_predict(_ctx("2020-03-11"))
    assert isinstance(d, QuantileDist)
    assert d.var(0.025) == -0.09 and d.es(0.01) == -0.17
    assert d.sigma2() == pytest.approx(0.0016)


def test_bridge_falls_back_when_date_missing():
    b = MSGarchBridge(pd.DataFrame([{
        "asset": "BTC", "prev_date": "2020-03-11", "date": "2020-03-12",
        "var_0025": -0.09, "es_0025": -0.12, "var_001": -0.14, "es_001": -0.17, "sigma2": 0.0016,
    }]))
    assert isinstance(b.fit_predict(_ctx("2021-01-01")), EmpiricalDist)      # unknown date
    assert isinstance(b.fit_predict(_ctx("2020-03-11", "ETH")), EmpiricalDist)  # unknown asset


def test_bridge_empty_cache_is_all_fallback():
    b = MSGarchBridge(None)
    assert isinstance(b.fit_predict(_ctx("2020-03-11")), EmpiricalDist)


def test_from_store_missing_db_is_safe(tmp_path):
    b = MSGarchBridge.from_store(tmp_path / "nope.duckdb")
    assert isinstance(b.fit_predict(_ctx("2020-03-11")), EmpiricalDist)
