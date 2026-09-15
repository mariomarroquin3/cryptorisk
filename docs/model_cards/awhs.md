# Model card: AWHS

**Family:** Non-parametric  
**Idea:** Historical Simulation with exponentially decaying weights (half-life 125d).

## Out-of-sample scorecard

| asset | a | hit rate | coverage | ES Z2 | ES ok | FZ0 rank | in MCS |
|:--|--:|--:|:--:|--:|:--:|--:|:--:|
| BTC | 0.01 | 0.011 | pass | -0.20 | yes | 13/18 | yes |
| BTC | 0.025 | 0.023 | pass | -0.00 | yes | 12/18 | yes |
| ETH | 0.01 | 0.014 | FAIL | -0.47 | no | 14/18 | yes |
| ETH | 0.025 | 0.025 | FAIL | -0.07 | yes | 14/18 | yes |

**Density (Berkowitz):** BTC p=0.145, ETH p=0.030 (reject)

**Volatility forecast (QLIKE):** BTC rank 11/15 (MZ b=1.23), ETH rank 12/15 (MZ b=1.08)

**Decision layer:** BTC capital $430,936 (m_c 1.50), N* $1,040,083; ETH capital $581,974 (m_c 1.50), N* $767,016

## Known limitations

- Effective sample size ~ H/ln2 << w, so the tail estimate is noisier.
- Adapts to the recency of the tail, not its shape.

_The estimator, likelihood and closed-form VaR/ES are in [`../methodology.tex`](../methodology.tex)._
