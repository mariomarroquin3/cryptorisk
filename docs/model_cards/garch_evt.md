# Model card: GARCH-EVT

**Family:** Semiparametric tail  
**Idea:** GARCH filter, then a Generalized Pareto fit to the residual tail (McNeil-Frey).

## Out-of-sample scorecard

| asset | a | hit rate | coverage | ES Z2 | ES ok | FZ0 rank | in MCS |
|:--|--:|--:|:--:|--:|:--:|--:|:--:|
| BTC | 0.01 | 0.013 | FAIL | -0.33 | yes | 16/20 | yes |
| BTC | 0.025 | 0.025 | pass | -0.04 | yes | 15/20 | yes |
| ETH | 0.01 | 0.009 | pass | -0.02 | yes | 7/20 | yes |
| ETH | 0.025 | 0.025 | FAIL | -0.04 | yes | 7/20 | yes |

**Density (Berkowitz):** BTC p=0.932, ETH p=0.704

**Volatility forecast (QLIKE):** BTC rank 9/17 (MZ b=0.96), ETH rank 9/17 (MZ b=0.71)

**Decision layer:** BTC capital $580,009 (m_c 1.90), N* $927,391; ETH capital $574,889 (m_c 1.50), N* $775,502

## Known limitations

- Sensitive to the threshold (fixed here at the 90th loss percentile).
- The GPD shape parameter is estimated from few exceedances.

_The estimator, likelihood and closed-form VaR/ES are in [`../methodology.tex`](../methodology.tex)._
