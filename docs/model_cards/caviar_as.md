# Model card: CAViaR-AS

**Family:** Exogenous / conditional  
**Idea:** Autoregression of the quantile itself (asymmetric slope), fit by the tick loss.

## Out-of-sample scorecard

| asset | a | hit rate | coverage | ES Z2 | ES ok | FZ0 rank | in MCS |
|:--|--:|--:|:--:|--:|:--:|--:|:--:|
| BTC | 0.01 | 0.011 | pass | -0.24 | yes | 11/18 | yes |
| BTC | 0.025 | 0.026 | pass | -0.12 | yes | 10/18 | yes |
| ETH | 0.01 | 0.012 | FAIL | -0.23 | yes | 8/18 | yes |
| ETH | 0.025 | 0.028 | FAIL | -0.16 | yes | 13/18 | yes |

**Density:** no full predictive density (quantile-only model).

**Decision layer:** BTC capital $535,728 (m_c 1.90), N* $1,088,694; ETH capital $561,264 (m_c 1.50), N* $777,809

## Known limitations

- No full density: no PIT / Berkowitz, and ES is a scaled quantile.
- Fit separately per alpha; cross-alpha monotonicity is repaired ex-post.

_The estimator, likelihood and closed-form VaR/ES are in [`../methodology.tex`](../methodology.tex)._
