# Model card: EGARCH-t

**Family:** Single-regime GARCH  
**Idea:** Models log-variance, so positivity is free and the sign asymmetry is direct.

## Out-of-sample scorecard

| asset | a | hit rate | coverage | ES Z2 | ES ok | FZ0 rank | in MCS |
|:--|--:|--:|:--:|--:|:--:|--:|:--:|
| BTC | 0.01 | 0.009 | pass | -0.11 | yes | 16/17 | yes |
| BTC | 0.025 | 0.023 | pass | -0.01 | yes | 17/17 | yes |
| ETH | 0.01 | 0.010 | pass | -0.01 | yes | 9/17 | yes |
| ETH | 0.025 | 0.027 | pass | -0.03 | yes | 9/17 | yes |

**Density (Berkowitz):** BTC p=0.898, ETH p=0.614

**Volatility forecast (QLIKE):** BTC rank 14/14 (MZ b=0.00), ETH rank 14/14 (MZ b=0.02)

**Decision layer:** BTC capital $519,209 (m_c 1.50), N* $786,973; ETH capital $621,567 (m_c 1.50), N* $676,818

## Known limitations

- The log-variance recursion can collapse toward 0 (guarded by a vol floor) or blow up near the non-stationary boundary (vol ceiling).
- Absorbs excess kurtosis into the recursion, leaving nu large and the 1% quantile thin.

_The estimator, likelihood and closed-form VaR/ES are in [`../methodology.tex`](../methodology.tex)._
