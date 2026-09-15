# Model card: GARCH-X

**Family:** Exogenous / conditional  
**Idea:** GARCH-t with lagged RV added to the variance equation.

## Out-of-sample scorecard

| asset | a | hit rate | coverage | ES Z2 | ES ok | FZ0 rank | in MCS |
|:--|--:|--:|:--:|--:|:--:|--:|:--:|
| BTC | 0.01 | 0.007 | pass | +0.25 | yes | 15/18 | yes |
| BTC | 0.025 | 0.019 | FAIL | +0.23 | yes | 14/18 | yes |
| ETH | 0.01 | 0.008 | pass | +0.18 | yes | 6/18 | yes |
| ETH | 0.025 | 0.026 | pass | +0.01 | yes | 5/18 | yes |

**Density (Berkowitz):** BTC p=0.000 (reject), ETH p=0.053

**Volatility forecast (QLIKE):** BTC rank 13/15 (MZ b=0.11), ETH rank 11/15 (MZ b=0.42)

**Decision layer:** BTC capital $549,688 (m_c 1.50), N* $763,906; ETH capital $618,146 (m_c 1.50), N* $686,656

## Known limitations

- Degrades to plain GARCH-t (gamma=0) when the realized block is absent.
- The exogenous term is a single contemporaneous lag.

_The estimator, likelihood and closed-form VaR/ES are in [`../methodology.tex`](../methodology.tex)._
