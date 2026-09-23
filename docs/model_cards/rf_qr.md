# Model card: RF-QR

**Family:** Machine learning  
**Idea:** Quantile Regression Forest (Meinshausen, 2006): a random forest whose leaves keep their full set of training returns instead of collapsing to a mean, so VaR/ES are weighted empirical quantiles of that pooled distribution -- no parametric tail assumption, unlike every GARCH-family model here. Features are HAR-style rolling means of squared returns (plus log RV when available), not a fitted variance recursion.

## Out-of-sample scorecard

| asset | a | hit rate | coverage | ES Z2 | ES ok | FZ0 rank | in MCS |
|:--|--:|--:|:--:|--:|:--:|--:|:--:|
| BTC | 0.01 | 0.010 | pass | -0.17 | yes | 10/20 | yes |
| BTC | 0.025 | 0.026 | pass | -0.12 | yes | 8/20 | yes |
| ETH | 0.01 | 0.013 | pass | -0.44 | yes | 14/20 | yes |
| ETH | 0.025 | 0.028 | FAIL | -0.18 | yes | 14/20 | yes |

**Density (Berkowitz):** BTC p=0.039 (reject), ETH p=0.001 (reject)

**Volatility forecast (QLIKE):** BTC rank 6/17 (MZ b=1.34), ETH rank 6/17 (MZ b=1.22)

**Decision layer:** BTC capital $522,414 (m_c 1.90), N* $1,116,003; ETH capital $557,127 (m_c 1.50), N* $822,443

## Known limitations

- No explicit time-series dynamics -- the forest sees lagged features, not a state that evolves (no persistence parameter like GARCH's alpha+beta).
- The tail is only as resolved as the leaves that survive to the test point; extreme quantiles rely on however few training returns land there.
- VaR and ES are built from returns observed in the window, so they can never be more extreme than the window's worst day: a new record loss is always a violation.

_The estimator, likelihood and closed-form VaR/ES are in [`../methodology.tex`](../methodology.tex)._
