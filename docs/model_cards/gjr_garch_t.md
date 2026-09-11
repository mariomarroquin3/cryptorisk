# Model card: GJR-GARCH-t

**Family:** Single-regime GARCH  
**Idea:** GARCH-t plus a leverage term (bad news raises variance more).

## Out-of-sample scorecard

| asset | a | hit rate | coverage | ES Z2 | ES ok | FZ0 rank | in MCS |
|:--|--:|--:|:--:|--:|:--:|--:|:--:|
| BTC | 0.01 | 0.011 | FAIL | -0.03 | yes | 8/17 | yes |
| BTC | 0.025 | 0.024 | pass | +0.04 | yes | 3/17 | yes |
| ETH | 0.01 | 0.009 | pass | +0.12 | yes | 3/17 | yes |
| ETH | 0.025 | 0.025 | FAIL | +0.01 | yes | 4/17 | yes |

**Density (Berkowitz):** BTC p=0.958, ETH p=0.808

**Volatility forecast (QLIKE):** BTC rank 9/14 (MZ b=0.90), ETH rank 9/14 (MZ b=0.75)

**Decision layer:** BTC capital $458,928 (m_c 1.50), N* $899,846; ETH capital $591,081 (m_c 1.50), N* $711,250

## Known limitations

- The asymmetry is a single extra parameter, constant over time.

_The estimator, likelihood and closed-form VaR/ES are in [`../methodology.tex`](../methodology.tex)._
