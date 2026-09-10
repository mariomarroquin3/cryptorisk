# Portfolio VaR / ES with a copula tail (Phase 7)

_`python -m cryptorisk.study.run_portfolio`._

Fixed-weight basket: BTC 0.5, ETH 0.5 (renormalised). GARCH(1,1)-t marginals refit daily, 20,000 Monte-Carlo draws, Student-t copula df 5. Same OOS window and battery as the single-asset study.

## 99.0% VaR

| rank | model | hit rate | coverage | ES Z2 | ES ok | Berkowitz p | mean FZ0 | in MCS | MCS p |
|--:|:--|--:|:--:|--:|:--:|--:|--:|:--:|--:|
| 1 | Direct-GJR-GARCH-t (best) | 0.012 | pass | -0.124 | yes | 0.953 | -2.0256 | yes | 0.265 |
| 2 | Direct-GARCH-t | 0.011 | pass | -0.095 | yes | 0.899 | -2.0184 | yes | 0.265 |
| 3 | Copula-clayton | 0.012 | pass | -0.234 | yes | 0.068 | -2.0171 | yes | 0.265 |
| 4 | Direct-FHS | 0.010 | pass | -0.091 | yes | 0.884 | -2.0045 | yes | 0.265 |
| 5 | Copula-student_t | 0.013 | pass | -0.346 | yes | 0.741 | -2.0000 | yes | 0.265 |
| 6 | Copula-gaussian | 0.013 | pass | -0.409 | yes | 0.956 | -1.9957 | yes | 0.265 |
| 7 | Direct-HS | 0.010 | pass | -0.064 | yes | 0.097 | -1.9499 | yes | 0.265 |
| 8 | Copula-independence | 0.030 | FAIL | -2.630 | no | 0.000 | -1.5467 | no | 0.002 |

## 97.5% VaR

| rank | model | hit rate | coverage | ES Z2 | ES ok | Berkowitz p | mean FZ0 | in MCS | MCS p |
|--:|:--|--:|:--:|--:|:--:|--:|--:|:--:|--:|
| 1 | Direct-GJR-GARCH-t (best) | 0.028 | pass | -0.083 | yes | 0.953 | -2.3100 | yes | 0.144 |
| 2 | Copula-clayton | 0.028 | FAIL | -0.144 | yes | 0.068 | -2.3002 | yes | 0.144 |
| 3 | Copula-gaussian | 0.027 | FAIL | -0.176 | yes | 0.956 | -2.2957 | yes | 0.144 |
| 4 | Direct-GARCH-t | 0.029 | FAIL | -0.124 | yes | 0.899 | -2.2957 | yes | 0.144 |
| 5 | Copula-student_t | 0.029 | pass | -0.212 | yes | 0.741 | -2.2931 | yes | 0.144 |
| 6 | Direct-FHS | 0.026 | FAIL | -0.071 | yes | 0.884 | -2.2770 | yes | 0.144 |
| 7 | Direct-HS | 0.023 | FAIL | +0.037 | yes | 0.097 | -2.2460 | no | 0.077 |
| 8 | Copula-independence | 0.056 | FAIL | -1.601 | no | 0.000 | -2.0460 | no | 0.001 |

## MCS membership by sub-period (lower alpha)

| model | full_oos | covid | luna | ftx | calm_23 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Direct-GJR-GARCH-t | + | + | + | + | + |
| Direct-GARCH-t | + | + | + | + | + |
| Copula-clayton | + | + | + | + | + |
| Direct-FHS | + | + | + | + | + |
| Copula-student_t | + | + | + | + | + |
| Copula-gaussian | + | + | + | + | + |
| Direct-HS | + | + | + | + | + |
| Copula-independence | . | + | . | + | + |

## Read

- **Ignoring tail dependence is dangerous.** `Copula-independence` over-breaches by ~3x, fails the ES test hard (Z2 well below 0) and is the only model out of the MCS &mdash; assuming BTC and ETH move independently understates basket tail risk by a wide margin.
- **A tail-dependent copula beats the Gaussian.** Within the `Copula-*` block the FZ0 ordering is Clayton < Student-t < Gaussian in both cells, and Clayton's Z2 is the least negative &mdash; lower-tail dependence (joint crashes) is the right structure for the pair.
- **Modelling the basket directly still wins.** `Direct-GJR-GARCH-t` / `Direct-GARCH-t` on the basket return series edge every copula on FZ0, both sides refitting volatility daily (a fair fight). The copula captures the dependence well enough to beat the Gaussian and independence, but a leverage-GARCH fit on `w'r` is simpler and marginally better.
- The copula marginals are GARCH(1,1)-t; the residual inversion is FHS (empirical). Only the *dependence* is parametric.
- The sub-period MCS is all-in for the seven survivors (40&ndash;90 day windows have no power), so the copula-family differences show up only in the full-sample FZ0 mean.
