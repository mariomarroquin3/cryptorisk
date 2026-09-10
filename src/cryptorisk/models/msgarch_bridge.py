"""MS-GARCH bridge (V2_PLAN §3, §5.5, Phase 2c).

MS-GARCH's walk-forward runs entirely in R (``msgarch/fit_msgarch_walkforward.R``,
driven by ``cryptorisk.study.run_msgarch``); the per-day predictions are cached
in the store's ``msgarch_predictions`` table. This model just serves them:
``fit_predict`` looks up ``(asset, asof)`` and returns a
:class:`~cryptorisk.models.base.QuantileDist`. Dates without a cached prediction
(no store, R not run, or a calendar gap) fall back to the empirical quantile.

The v1 finding stands (V2_PLAN §5.5): the walk-forward regime probability
(``prob_crisis_filt`` / ``prob_crisis_pred``) barely tracks volatility; only the
single full-sample fit's ``prob_crisis_insample`` does. Those columns are served
for the regime study, not used in the VaR forecast here.
"""

from __future__ import annotations

import pandas as pd

from cryptorisk.models.base import Context, EmpiricalDist, PredictiveDist, QuantileDist


class MSGarchBridge:
    name = "MS-GARCH"

    def __init__(self, cache: pd.DataFrame | None = None, alphas=(0.025, 0.01)):
        self.alphas = tuple(alphas)
        self._by_key: dict[tuple[str, pd.Timestamp], dict] = {}
        if cache is not None and len(cache):
            for row in cache.to_dict("records"):
                key = (str(row["asset"]), pd.Timestamp(row["prev_date"]).normalize())
                self._by_key[key] = row

    @classmethod
    def from_store(cls, db, alphas=(0.025, 0.01)) -> MSGarchBridge:
        try:
            import duckdb

            con = duckdb.connect(str(db), read_only=True)
            df = con.execute("SELECT * FROM msgarch_predictions").df()
            con.close()
        except Exception:  # noqa: BLE001 - no store / no table / no duckdb
            df = None
        return cls(df, alphas=alphas)

    def fit_predict(self, ctx: Context) -> PredictiveDist:
        row = self._by_key.get((ctx.asset, pd.Timestamp(ctx.asof).normalize()))
        if row is None:
            return EmpiricalDist(ctx.returns)
        var = {0.025: float(row["var_0025"]), 0.01: float(row["var_001"])}
        es = {0.025: float(row["es_0025"]), 0.01: float(row["es_001"])}
        s2 = row.get("sigma2")
        return QuantileDist(var, es, _sigma2=float(s2) if s2 == s2 else None)
