# Model card: HS

**Family:** Non-parametric  
**Idea:** Empirical quantile of the last w returns; no distributional assumption.

## Out-of-sample scorecard

| asset | a | hit rate | coverage | ES Z2 | ES ok | FZ0 rank | in MCS |
|:--|--:|--:|:--:|--:|:--:|--:|:--:|
| BTC | 0.01 | 0.010 | pass | -0.02 | yes | 6/16 | yes |
| BTC | 0.025 | 0.023 | pass | +0.05 | yes | 7/16 | yes |
| ETH | 0.01 | 0.012 | FAIL | -0.21 | yes | 14/16 | yes |
| ETH | 0.025 | 0.025 | FAIL | -0.04 | yes | 14/16 | no |

**Density (Berkowitz):** BTC p=0.212, ETH p=0.098

**Volatility forecast (QLIKE):** BTC rank 10/14 (MZ b=1.23), ETH rank 11/14 (MZ b=1.08)

**Decision layer:** BTC capital $445,654 (m_c 1.50), N* $985,070; ETH capital $602,729 (m_c 1.50), N* $722,694

## Known limitations

- Reacts slowly to a regime change (the whole window must roll over).
- The tail is only as resolved as `alpha*w` observations.

_The estimator, likelihood and closed-form VaR/ES are in [`../methodology.tex`](../methodology.tex)._
