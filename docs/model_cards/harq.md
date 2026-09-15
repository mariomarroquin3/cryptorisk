# Model card: HARQ

**Family:** Realized-measure  
**Idea:** HAR-RV with the daily lag shrunk on days when RV was measured noisily.

## Out-of-sample scorecard

| asset | a | hit rate | coverage | ES Z2 | ES ok | FZ0 rank | in MCS |
|:--|--:|--:|:--:|--:|:--:|--:|:--:|
| BTC | 0.01 | 0.023 | FAIL | -1.96 | no | 18/18 | yes |
| BTC | 0.025 | 0.035 | FAIL | -0.79 | no | 17/18 | yes |
| ETH | 0.01 | 0.023 | FAIL | -1.72 | no | 16/18 | yes |
| ETH | 0.025 | 0.036 | FAIL | -0.73 | no | 12/18 | yes |

**Density (Berkowitz):** BTC p=0.000 (reject), ETH p=0.000 (reject)

**Volatility forecast (QLIKE):** BTC rank 2/15 (MZ b=0.61), ETH rank 1/15 (MZ b=0.67)

**Decision layer:** BTC capital $430,177 (m_c 1.90), N* $1,457,809; ETH capital $436,260 (m_c 1.50), N* $1,135,093

## Known limitations

- Best variance forecaster, near-worst for VaR/ES: the tail is a point forecast with a thin t on top.
- The RQ interaction is z-scored using window-wide moments (mild leak).

_The estimator, likelihood and closed-form VaR/ES are in [`../methodology.tex`](../methodology.tex)._
