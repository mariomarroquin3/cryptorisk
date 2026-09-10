"""Phase 7 (extension): multi-asset portfolio VaR / ES with a copula for the
joint tail (V2_PLAN §7).

A fixed-weight BTC + ETH basket. Two ways to get its 1-day VaR / ES:

* **direct** -- run the univariate engine on the basket log-return series
  (any model in ``registry``);
* **copula** -- filter each asset's volatility (GARCH(1,1)-t), fit a copula to
  the standardized-residual pseudo-observations, Monte-Carlo simulate the joint
  standardized shocks, scale by each asset's one-step sigma and aggregate.

Copula families: independence, Gaussian, Student-t (symmetric tail
dependence), Clayton (lower-tail dependence). The comparison is the point:
does modelling the dependence beat modelling the basket directly, and does a
tail-dependent copula beat the Gaussian?

Evaluated with the same battery as the single-asset study (coverage, ES,
FZ0 + Model Confidence Set, Berkowitz, sub-periods).
"""
