# Model card: LSTM-Vol

**Family:** Machine learning  
**Idea:** An LSTM reads the last 20 days of (return, squared return, squared down-return) and outputs next-day log-variance, trained by Gaussian quasi-MLE -- the same quasi-MLE principle as the GARCH family, with a gated recurrent net standing in for the fixed GARCH(1,1) recursion. VaR/ES come from a Student-t tail fitted to the in-sample standardized residuals, same as HAR-RV.

## Out-of-sample scorecard

| asset | a | hit rate | coverage | ES Z2 | ES ok | FZ0 rank | in MCS |
|:--|--:|--:|:--:|--:|:--:|--:|:--:|
| BTC | 0.01 | 0.012 | FAIL | -0.12 | yes | 8/20 | yes |
| BTC | 0.025 | 0.029 | FAIL | -0.13 | yes | 14/20 | yes |
| ETH | 0.01 | 0.017 | FAIL | -0.65 | no | 14/20 | yes |
| ETH | 0.025 | 0.035 | FAIL | -0.41 | no | 19/20 | yes |

**Density (Berkowitz):** BTC p=0.024 (reject), ETH p=0.000 (reject)

**Volatility forecast (QLIKE):** BTC rank 12/17 (MZ b=0.73), ETH rank 12/17 (MZ b=0.59)

**Decision layer:** BTC capital $442,435 (m_c 1.50), N* $947,785; ETH capital $539,064 (m_c 1.50), N* $791,747

## Known limitations

- Refits only every 20 days (gradient descent, not a closed form or a handful of L-BFGS-B steps) -- coarser than every other model here, which refit daily.
- A single fixed architecture/seed per window: no hyperparameter search, no ensembling across initializations.

_The estimator, likelihood and closed-form VaR/ES are in [`../methodology.tex`](../methodology.tex)._
