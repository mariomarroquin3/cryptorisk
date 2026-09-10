# Model card: Realized-GARCH

**Family:** Realized-measure  
**Idea:** GARCH driven by observed RV, with a measurement equation tying the two (Hansen-Huang-Shek).

## Out-of-sample scorecard

| asset | a | hit rate | coverage | ES Z2 | ES ok | FZ0 rank | in MCS |
|:--|--:|--:|:--:|--:|:--:|--:|:--:|
| BTC | 0.01 | 0.010 | pass | +0.02 | yes | 1/16 | yes |
| BTC | 0.025 | 0.028 | pass | -0.06 | yes | 1/16 | yes |
| ETH | 0.01 | 0.012 | pass | -0.14 | yes | 1/16 | yes |
| ETH | 0.025 | 0.030 | pass | -0.16 | yes | 2/16 | yes |

**Density (Berkowitz):** BTC p=0.223, ETH p=0.115

**Volatility forecast (QLIKE):** BTC rank 4/14 (MZ b=0.77), ETH rank 4/14 (MZ b=0.62)

**Decision layer:** BTC capital $443,045 (m_c 1.50), N* $949,269; ETH capital $567,477 (m_c 1.50), N* $748,858

## Known limitations

- Eight parameters -- more to estimate than plain GARCH.
- Assumes a Gaussian measurement error for log RV.

_The estimator, likelihood and closed-form VaR/ES are in [`../methodology.tex`](../methodology.tex)._
