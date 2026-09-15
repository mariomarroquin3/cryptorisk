# Model card: CAViaR-SAV

**Family:** Exogenous / conditional  
**Idea:** Autoregression of the quantile itself (symmetric absolute value), fit by the tick loss.

## Out-of-sample scorecard

| asset | a | hit rate | coverage | ES Z2 | ES ok | FZ0 rank | in MCS |
|:--|--:|--:|:--:|--:|:--:|--:|:--:|
| BTC | 0.01 | 0.011 | pass | -0.16 | yes | 3/18 | yes |
| BTC | 0.025 | 0.026 | pass | -0.14 | yes | 15/18 | yes |
| ETH | 0.01 | 0.015 | FAIL | -0.59 | no | 17/18 | yes |
| ETH | 0.025 | 0.026 | FAIL | -0.13 | yes | 18/18 | yes |

**Density:** no full predictive density (quantile-only model).

**Decision layer:** BTC capital $541,048 (m_c 1.90), N* $1,075,111; ETH capital $579,856 (m_c 1.50), N* $776,454

## Known limitations

- No full density: no PIT / Berkowitz, and ES is a scaled quantile.
- Symmetric news impact -- unlike AS it cannot react differently to good vs bad days.
- Fit separately per alpha; cross-alpha monotonicity is repaired ex-post.

_The estimator, likelihood and closed-form VaR/ES are in [`../methodology.tex`](../methodology.tex)._
