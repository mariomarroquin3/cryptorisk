# Model card: CAViaR-X-AS

**Family:** Exogenous / conditional  
**Idea:** CAViaR-AS plus sqrt(RV_{t-1}) as an exogenous quantile driver.

## Out-of-sample scorecard

| asset | a | hit rate | coverage | ES Z2 | ES ok | FZ0 rank | in MCS |
|:--|--:|--:|:--:|--:|:--:|--:|:--:|
| BTC | 0.01 | 0.011 | pass | -0.21 | yes | 5/18 | yes |
| BTC | 0.025 | 0.025 | pass | -0.08 | yes | 11/18 | yes |
| ETH | 0.01 | 0.012 | pass | -0.22 | yes | 5/18 | yes |
| ETH | 0.025 | 0.028 | FAIL | -0.15 | yes | 6/18 | yes |

**Density:** no full predictive density (quantile-only model).

**Decision layer:** BTC capital $421,255 (m_c 1.50), N* $1,102,366; ETH capital $560,597 (m_c 1.50), N* $784,559

## Known limitations

- Same density / ES caveats as CAViaR-AS.
- Adds one parameter estimated by a derivative-free search.

_The estimator, likelihood and closed-form VaR/ES are in [`../methodology.tex`](../methodology.tex)._
