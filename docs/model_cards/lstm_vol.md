# Model card: LSTM-Vol

**Family:** Machine learning  
**Idea:** An LSTM reads the last 20 days of (return, squared return, squared down-return) and outputs next-day log-variance, trained by Gaussian quasi-MLE -- the same quasi-MLE principle as the GARCH family, with a gated recurrent net standing in for the fixed GARCH(1,1) recursion. VaR/ES come from a Student-t tail fitted to the in-sample standardized residuals, same as HAR-RV.

## Out-of-sample scorecard

| asset | a | hit rate | coverage | ES Z2 | ES ok | FZ0 rank | in MCS |
|:--|--:|--:|:--:|--:|:--:|--:|:--:|
| BTC | 0.01 | 0.013 | pass | -0.19 | yes | 7/20 | yes |
| BTC | 0.025 | 0.030 | FAIL | -0.14 | yes | 13/20 | yes |
| ETH | 0.01 | 0.015 | FAIL | -0.37 | yes | 13/20 | yes |
| ETH | 0.025 | 0.031 | FAIL | -0.25 | no | 15/20 | yes |

**Density (Berkowitz):** BTC p=0.027 (reject), ETH p=0.004 (reject)

**Volatility forecast (QLIKE):** BTC rank 12/17 (MZ b=0.75), ETH rank 13/17 (MZ b=0.88)

**Decision layer:** BTC capital $439,844 (m_c 1.50), N* $953,675; ETH capital $551,978 (m_c 1.50), N* $773,673

## Known limitations

- The weights are retrained only every 20 days (gradient descent, not a closed form or a handful of L-BFGS-B steps); between retrains the stored network still runs daily on the newest 20 returns, but its parameters are up to 19 days stale -- coarser than every other model here, which refit daily.
- The inputs are raw returns (squared returns are ~1e-4), so permutation importance puts ~94% on the signed return: the network barely uses its squared-return channels. Standardizing them was tried and made calibration worse (hit rates ~4-6% at a 2.5% target), so this is an open modelling limitation.
- A single fixed architecture/seed per window: no hyperparameter search, no ensembling across initializations.

_The estimator, likelihood and closed-form VaR/ES are in [`../methodology.tex`](../methodology.tex)._
