# Model card: HAR-RV

**Family:** Realized-measure  
**Idea:** OLS of realized variance on its daily / weekly / monthly averages (Corsi).

## Out-of-sample scorecard

| asset | a | hit rate | coverage | ES Z2 | ES ok | FZ0 rank | in MCS |
|:--|--:|--:|:--:|--:|:--:|--:|:--:|
| BTC | 0.01 | 0.016 | FAIL | -0.88 | no | 5/17 | yes |
| BTC | 0.025 | 0.027 | pass | -0.25 | yes | 2/17 | yes |
| ETH | 0.01 | 0.019 | FAIL | -1.21 | no | 8/17 | yes |
| ETH | 0.025 | 0.030 | pass | -0.41 | no | 1/17 | yes |

**Density (Berkowitz):** BTC p=0.011 (reject), ETH p=0.585

**Volatility forecast (QLIKE):** BTC rank 3/14 (MZ b=1.44), ETH rank 2/14 (MZ b=1.20)

**Decision layer:** BTC capital $352,057 (m_c 1.50), N* $1,406,302; ETH capital $567,305 (m_c 1.90), N* $1,105,665

## Known limitations

- Targets the conditional *mean* of RV; RV is right-skewed, so the model runs thin for a lower return quantile.
- Linear -- no volatility-of-volatility term.

_The estimator, likelihood and closed-form VaR/ES are in [`../methodology.tex`](../methodology.tex)._
