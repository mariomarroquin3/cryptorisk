# Model card: MS-GARCH

**Family:** Regime-switching  
**Idea:** Two-regime Markov-switching GARCH mixture, walk-forward in R.

## Out-of-sample scorecard

| asset | a | hit rate | coverage | ES Z2 | ES ok | FZ0 rank | in MCS |
|:--|--:|--:|:--:|--:|:--:|--:|:--:|
| BTC | 0.01 | 0.010 | pass | -0.23 | yes | 10/16 | yes |
| BTC | 0.025 | 0.023 | pass | -0.01 | yes | 5/16 | yes |
| ETH | 0.01 | 0.011 | FAIL | -0.25 | yes | 10/16 | yes |
| ETH | 0.025 | 0.024 | pass | -0.03 | yes | 8/16 | yes |

**Density:** no full predictive density (quantile-only model).

**Volatility forecast (QLIKE):** BTC rank 6/14 (MZ b=1.65), ETH rank 6/14 (MZ b=1.42)

**Decision layer:** BTC capital $412,569 (m_c 1.50), N* $1,120,608; ETH capital $533,910 (m_c 1.50), N* $862,531

## Known limitations

- The walk-forward regime probability has ~no out-of-sample signal (see the regime-identification section); the layer is descriptive.
- Served from a cache; dates without a prediction fall back to empirical.

_The estimator, likelihood and closed-form VaR/ES are in [`../methodology.tex`](../methodology.tex)._
