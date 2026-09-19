# Model card: Realized-GARCH

**Family:** Realized-measure  
**Idea:** GARCH driven by observed RV, with a measurement equation tying the two (Hansen-Huang-Shek).

## Out-of-sample scorecard

| asset | a | hit rate | coverage | ES Z2 | ES ok | FZ0 rank | in MCS |
|:--|--:|--:|:--:|--:|:--:|--:|:--:|
| BTC | 0.01 | 0.010 | pass | +0.02 | yes | 2/20 | yes |
| BTC | 0.025 | 0.028 | pass | -0.06 | yes | 2/20 | yes |
| ETH | 0.01 | 0.012 | pass | -0.14 | yes | 1/20 | yes |
| ETH | 0.025 | 0.030 | pass | -0.16 | yes | 2/20 | yes |

**Density (Berkowitz):** BTC p=0.221, ETH p=0.116

**Volatility forecast (QLIKE):** BTC rank 5/17 (MZ b=0.77), ETH rank 5/17 (MZ b=0.62)

**Decision layer:** BTC capital $442,974 (m_c 1.50), N* $949,411; ETH capital $567,459 (m_c 1.50), N* $748,883

## Known limitations

- Eight parameters -- more to estimate than plain GARCH.
- Assumes a Gaussian measurement error for log RV.

_The estimator, likelihood and closed-form VaR/ES are in [`../methodology.tex`](../methodology.tex)._
