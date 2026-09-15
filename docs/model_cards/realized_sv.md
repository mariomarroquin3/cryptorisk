# Model card: Realized-SV

**Family:** Stochastic volatility  
**Idea:** Heston-style CIR variance filtered with an Unscented Kalman Filter from the realized measure alone (log RV_t = zeta + phi*log V_t + u_t, phi fitted not fixed at 1), plus a GJR-style lagged leverage term. The return does not update the filtered state -- it scores its own exact Student-t log-density given the predicted V instead of going through the approximate Harvey-Ruiz-Shephard log-square trick tried during development, which stayed biased even after correcting its Gaussian constants for fat tails. VaR/ES come from a GPD tail (McNeil-Frey, 2000, same construction as GARCH-EVT) fitted to the filtered window's own standardized residuals, not the fitted Student-t directly -- a single nu from the whole-sample likelihood undersold the most extreme moves.

## Out-of-sample scorecard

| asset | a | hit rate | coverage | ES Z2 | ES ok | FZ0 rank | in MCS |
|:--|--:|--:|:--:|--:|:--:|--:|:--:|
| BTC | 0.01 | 0.011 | FAIL | -0.08 | yes | 1/18 | yes |
| BTC | 0.025 | 0.025 | pass | +0.01 | yes | 1/18 | yes |
| ETH | 0.01 | 0.010 | pass | -0.11 | yes | 4/18 | yes |
| ETH | 0.025 | 0.025 | pass | -0.02 | yes | 8/18 | yes |

**Density (Berkowitz):** BTC p=0.927, ETH p=0.809

**Volatility forecast (QLIKE):** BTC rank 1/15 (MZ b=1.21), ETH rank 3/15 (MZ b=1.50)

## Known limitations

- Leverage is lagged (gamma * I(r_{t-1}<0) * resid_{t-1}^2 in the CIR drift), not Heston's same-day correlated shocks (rho != 0) -- avoids a same-day causality problem in the filter, at the cost of a one-day-slower asymmetric response.
- Quasi-MLE via an approximate filter likelihood (Gaussian on the RV channel only), not the exact CIR transition density.

_The estimator, likelihood and closed-form VaR/ES are in [`../methodology.tex`](../methodology.tex)._
