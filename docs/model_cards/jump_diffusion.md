# Model card: Jump-Diffusion

**Family:** Discontinuous  
**Idea:** Brownian part plus a compound-Poisson jump component (Merton).

## Out-of-sample scorecard

| asset | a | hit rate | coverage | ES Z2 | ES ok | FZ0 rank | in MCS |
|:--|--:|--:|:--:|--:|:--:|--:|:--:|
| BTC | 0.01 | 0.012 | pass | -0.19 | yes | 6/17 | yes |
| BTC | 0.025 | 0.024 | pass | -0.01 | yes | 8/17 | yes |
| ETH | 0.01 | 0.013 | FAIL | -0.37 | yes | 12/17 | yes |
| ETH | 0.025 | 0.024 | FAIL | -0.01 | yes | 15/17 | no |

**Density (Berkowitz):** BTC p=0.000 (reject), ETH p=0.000 (reject)

**Volatility forecast (QLIKE):** BTC rank 13/14 (MZ b=0.99), ETH rank 13/14 (MZ b=0.87)

**Decision layer:** BTC capital $439,183 (m_c 1.50), N* $1,000,905; ETH capital $600,026 (m_c 1.50), N* $724,453

## Known limitations

- Jump intensity and size are constant -- no clustering of jumps.
- VaR/ES come from a fixed-seed Monte-Carlo sample, so a small quantisation error remains.

_The estimator, likelihood and closed-form VaR/ES are in [`../methodology.tex`](../methodology.tex)._
