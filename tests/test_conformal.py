import numpy as np
import pandas as pd
from scipy.stats import norm

from cryptorisk.models.base import Context, PredictiveDist
from cryptorisk.models.conformal import AdaptiveConformal, AdjustedDist

ALPHAS = (0.025, 0.01)


class GaussDist(PredictiveDist):
    def __init__(self, sigma: float):
        self.s = sigma

    def var(self, alpha):
        return float(norm.ppf(alpha) * self.s)

    def es(self, alpha):
        return float(-self.s * norm.pdf(norm.ppf(alpha)) / alpha)

    def sigma2(self):
        return self.s**2

    def cdf(self, x):
        return float(norm.cdf(x / self.s))

    def ppf(self, u):
        return float(norm.ppf(u) * self.s)


class FixedScale:
    """A model that always believes the volatility is ``sigma`` (wrong on purpose)."""

    def __init__(self, sigma: float, name: str = "Fixed"):
        self.sigma, self.name = sigma, name

    def fit_predict(self, ctx):
        return GaussDist(self.sigma)


def _ctx(returns, day):
    d = np.datetime64("2020-01-01") + np.timedelta64(day, "D")
    n = len(returns)
    dates = d - np.arange(n)[::-1].astype("timedelta64[D]")
    return Context(returns=np.asarray(returns, float), dates=dates, asof=d, asset="BTC")


def test_adjusted_dist_reads_the_base_at_the_adjusted_level():
    base = GaussDist(0.02)
    adj = AdjustedDist(base, {0.025: 0.005})
    assert adj.var(0.025) == base.var(0.005) and adj.es(0.025) == base.es(0.005)
    assert adj.var(0.01) == base.var(0.01)            # a level with no adjustment passes through
    assert adj.cdf(-0.03) == base.cdf(-0.03) and adj.sigma2() == base.sigma2()


def test_breach_lowers_the_level_and_a_quiet_day_raises_it_slightly():
    m = AdaptiveConformal(FixedScale(0.02), ALPHAS)
    m.fit_predict(_ctx([0.0, 0.0], 1))
    var1 = m._state["BTC"].last_var[0.025]
    m.fit_predict(_ctx([0.0, var1 - 0.01], 2))         # yesterday's return breached the VaR
    breached = m._state["BTC"].levels[0.025]
    assert breached < 0.025
    m.fit_predict(_ctx([0.0, 0.0], 3))                 # a calm day
    assert m._state["BTC"].levels[0.025] > breached
    assert m._state["BTC"].levels[0.025] < 0.025 + 1e-9


def test_update_is_idempotent_and_skips_gaps():
    m = AdaptiveConformal(FixedScale(0.02), ALPHAS)
    m.fit_predict(_ctx([0.0, 0.0], 1))
    m.fit_predict(_ctx([0.0, -0.5], 2))
    lvl = m._state["BTC"].levels[0.025]
    m.fit_predict(_ctx([0.0, -0.5], 2))                # same asof again: no second update
    assert m._state["BTC"].levels[0.025] == lvl
    m.fit_predict(_ctx([0.0, -0.5], 9))                # a gap: the last forecast is not for this return
    assert m._state["BTC"].levels[0.025] == lvl


def test_level_is_clipped():
    m = AdaptiveConformal(FixedScale(0.02), ALPHAS, gamma_frac=50.0)
    m.fit_predict(_ctx([0.0, 0.0], 1))
    for day in range(2, 40):                            # breach every day, absurdly large steps
        m.fit_predict(_ctx([0.0, -1.0], day))
    assert m._state["BTC"].levels[0.025] >= 0.025 / 20 - 1e-12


def test_aci_repairs_a_model_that_underestimates_volatility():
    rng = np.random.default_rng(11)
    true_sigma, n = 0.02, 6000
    r = rng.normal(0, true_sigma, n)
    raw = FixedScale(0.8 * true_sigma)                  # believes vol is 20% lower than it is
    aci = AdaptiveConformal(FixedScale(0.8 * true_sigma), ALPHAS)
    hits = {"raw": [], "aci": []}
    for t in range(1, n):
        ctx = _ctx(r[: t + 1], t)                       # window ends at r[t]; forecast is for r[t+1]
        if t + 1 >= n:
            break
        hits["raw"].append(r[t + 1] < raw.fit_predict(ctx).var(0.025))
        hits["aci"].append(r[t + 1] < aci.fit_predict(ctx).var(0.025))
    raw_rate, aci_rate = np.mean(hits["raw"]), np.mean(hits["aci"])
    assert raw_rate > 0.05                              # over-violating: about 5.8% instead of 2.5%
    assert abs(aci_rate - 0.025) < 0.008                # repaired to ~nominal
    assert abs(aci_rate - 0.025) < abs(raw_rate - 0.025) / 3
    assert pd.Series([lv for *_, a, lv in aci.trace if a == 0.025]).iloc[-1] < 0.02  # by asking for a deeper quantile


def _bt(model, hits, alpha=0.025, asset="BTC"):
    n = len(hits)
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.02, n)
    var = np.where(hits, r + 0.001, r - 0.05)          # hit = the return is below the VaR
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n), "asset": asset, "model": model, "window": 500,
        "alpha": alpha, "var": var, "es": var * 1.3, "realized": r, "violation": r < var,
    })


def test_summarise_pairs_raw_and_aci_on_the_same_days():
    from cryptorisk.study.run_conformal import summarise

    n = 800
    rng = np.random.default_rng(1)
    raw_hits = rng.random(n) < 0.10                    # over-violating
    aci_hits = rng.random(n) < 0.025                   # calibrated
    raw = _bt("RF-QR", raw_hits)
    aci = _bt("RF-QR+ACI", aci_hits)
    out = summarise(raw, aci)
    assert list(out["variant"]) == ["raw", "ACI"] and set(out["base_model"]) == {"RF-QR"}
    r, a = out.iloc[0], out.iloc[1]
    assert r["n"] == a["n"] == n
    assert r["hit_rate"] > a["hit_rate"] and r["kupiec_p"] < 0.05 < a["kupiec_p"]
    assert np.isnan(r.get("dm_p", np.nan)) and np.isfinite(a["dm_p"])


def test_conformal_endpoints_filter_by_asset_and_alpha(monkeypatch):
    from cryptorisk.api import app as A
    from cryptorisk.api import data as D

    summary = pd.DataFrame([
        {"asset": "BTC", "alpha": 0.025, "base_model": "RF-QR", "variant": "raw", "hit_rate": 0.1},
        {"asset": "BTC", "alpha": 0.025, "base_model": "RF-QR", "variant": "ACI", "hit_rate": 0.03},
        {"asset": "ETH", "alpha": 0.025, "base_model": "RF-QR", "variant": "raw", "hit_rate": 0.2},
    ])
    path = pd.DataFrame([{"asset": "BTC", "alpha": 0.025, "date": "2024-01-01", "model": "RF-QR+ACI", "level": 0.02}])
    monkeypatch.setattr(D, "load_results", lambda: {"conformal_summary": summary, "conformal_path": path})
    assert [r["variant"] for r in A.conformal_summary("BTC", 0.025)] == ["raw", "ACI"]
    assert A.conformal_summary("ETH", 0.025)[0]["hit_rate"] == 0.2
    assert A.conformal_level_path("BTC", 0.025)[0]["level"] == 0.02
    monkeypatch.setattr(D, "load_results", lambda: {"conformal_summary": pd.DataFrame(), "conformal_path": pd.DataFrame()})
    assert A.conformal_summary("BTC", 0.025) == [] and A.conformal_level_path("BTC", 0.025) == []
