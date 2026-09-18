# Model card: GARCH-t

**Family:** Single-regime GARCH  
**Idea:** GARCH(1,1) variance recursion, standardized Student-t innovations.

## Out-of-sample scorecard

| asset | a | hit rate | coverage | ES Z2 | ES ok | FZ0 rank | in MCS |
|:--|--:|--:|:--:|--:|:--:|--:|:--:|
| BTC | 0.01 | 0.011 | FAIL | -0.06 | yes | 12/20 | yes |
| BTC | 0.025 | 0.026 | pass | -0.03 | yes | 7/20 | yes |
| ETH | 0.01 | 0.010 | pass | +0.07 | yes | 2/20 | yes |
| ETH | 0.025 | 0.028 | FAIL | -0.07 | yes | 3/20 | yes |

**Density (Berkowitz):** BTC p=0.972, ETH p=0.784

**Volatility forecast (QLIKE):** BTC rank 8/17 (MZ b=0.96), ETH rank 8/17 (MZ b=0.71)

**Decision layer:** BTC capital $459,751 (m_c 1.50), N* $895,601; ETH capital $589,056 (m_c 1.50), N* $712,339

## Known limitations

- Symmetric response to good and bad news.
- One regime: a single (alpha, beta) for calm and crisis alike.

_The estimator, likelihood and closed-form VaR/ES are in [`../methodology.tex`](../methodology.tex)._
