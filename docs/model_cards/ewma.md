# Model card: EWMA

**Family:** Exponential smoothing  
**Idea:** RiskMetrics: one-parameter recursive variance, Gaussian (or fixed-nu t) tail.

## Out-of-sample scorecard

| asset | a | hit rate | coverage | ES Z2 | ES ok | FZ0 rank | in MCS |
|:--|--:|--:|:--:|--:|:--:|--:|:--:|
| BTC | 0.01 | 0.020 | FAIL | -1.65 | no | 16/18 | yes |
| BTC | 0.025 | 0.029 | pass | -0.54 | no | 16/18 | yes |
| ETH | 0.01 | 0.024 | FAIL | -1.90 | no | 18/18 | yes |
| ETH | 0.025 | 0.034 | FAIL | -0.69 | no | 17/18 | yes |

**Density (Berkowitz):** BTC p=0.001 (reject), ETH p=0.008 (reject)

**Volatility forecast (QLIKE):** BTC rank 3/15 (MZ b=1.10), ETH rank 4/15 (MZ b=0.87)

**Decision layer:** BTC capital $421,629 (m_c 1.90), N* $1,499,960; ETH capital $554,878 (m_c 1.90), N* $1,139,758

## Known limitations

- Integrated (alpha+beta=1): no mean reversion, no finite unconditional variance.
- The Gaussian tail is too thin at alpha=0.01 -- see the ES test.

_The estimator, likelihood and closed-form VaR/ES are in [`../methodology.tex`](../methodology.tex)._
