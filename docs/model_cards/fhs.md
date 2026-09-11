# Model card: FHS

**Family:** Semiparametric tail  
**Idea:** GARCH filter, then the empirical tail of the standardized residuals.

## Out-of-sample scorecard

| asset | a | hit rate | coverage | ES Z2 | ES ok | FZ0 rank | in MCS |
|:--|--:|--:|:--:|--:|:--:|--:|:--:|
| BTC | 0.01 | 0.009 | pass | +0.01 | yes | 3/17 | yes |
| BTC | 0.025 | 0.024 | pass | -0.01 | yes | 4/17 | yes |
| ETH | 0.01 | 0.010 | FAIL | -0.14 | yes | 11/17 | yes |
| ETH | 0.025 | 0.025 | FAIL | -0.05 | yes | 10/17 | yes |

**Density (Berkowitz):** BTC p=0.867, ETH p=0.648

**Volatility forecast (QLIKE):** BTC rank 5/14 (MZ b=1.24), ETH rank 5/14 (MZ b=0.93)

**Decision layer:** BTC capital $432,358 (m_c 1.50), N* $1,021,097; ETH capital $570,918 (m_c 1.50), N* $773,430

## Known limitations

- Inherits any misspecification of the GARCH volatility dynamics.
- The residual tail is still a finite sample -- sparse at alpha=0.01.

_The estimator, likelihood and closed-form VaR/ES are in [`../methodology.tex`](../methodology.tex)._
