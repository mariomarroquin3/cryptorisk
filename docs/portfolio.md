# Portfolio VaR / ES with a copula tail (4-asset basket, Phase 7)

_`python -m cryptorisk.study.run_portfolio`._

Fixed-weight basket: BTC 0.25, ETH 0.25, SOL 0.25, BNB 0.25 (renormalised). GARCH(1,1)-t marginals refit daily, 20,000 Monte-Carlo draws, 4-dimensional copula (Student-t df 5). Same evaluation battery as the single-asset study; the joint out-of-sample window is bounded by the shortest series (SOL, Binance spot from 2020-08).

## 99.0% VaR

| rank | model | hit rate | coverage | ES Z2 | ES ok | Berkowitz p | mean FZ0 | in MCS | MCS p |
|--:|:--|--:|:--:|--:|:--:|--:|--:|:--:|--:|
| 1 | Direct-GJR-GARCH-t (best) | 0.013 | pass | -0.226 | yes | 0.145 | -2.2144 | yes | 0.696 |
| 2 | Direct-GARCH-t | 0.013 | pass | -0.276 | yes | 0.097 | -2.1931 | yes | 0.696 |
| 3 | Direct-FHS | 0.010 | pass | -0.022 | yes | 0.776 | -2.1813 | yes | 0.696 |
| 4 | Copula-clayton | 0.009 | pass | +0.040 | yes | 0.160 | -2.1808 | yes | 0.696 |
| 5 | Copula-student_t | 0.012 | pass | -0.236 | yes | 0.702 | -2.1698 | yes | 0.696 |
| 6 | Copula-gaussian | 0.012 | pass | -0.360 | yes | 0.663 | -2.1633 | yes | 0.696 |
| 7 | Direct-HS | 0.009 | pass | +0.076 | yes | 0.144 | -2.0583 | no | 0.010 |
| 8 | Copula-independence | 0.065 | FAIL | -7.414 | no | 0.000 | -0.1099 | no | 0.000 |

## 97.5% VaR

| rank | model | hit rate | coverage | ES Z2 | ES ok | Berkowitz p | mean FZ0 | in MCS | MCS p |
|--:|:--|--:|:--:|--:|:--:|--:|--:|:--:|--:|
| 1 | Direct-GJR-GARCH-t (best) | 0.028 | pass | -0.123 | yes | 0.145 | -2.4378 | yes | 0.576 |
| 2 | Copula-clayton | 0.025 | pass | +0.021 | yes | 0.160 | -2.4286 | yes | 0.576 |
| 3 | Direct-GARCH-t | 0.030 | pass | -0.182 | yes | 0.097 | -2.4232 | yes | 0.576 |
| 4 | Copula-student_t | 0.032 | pass | -0.278 | yes | 0.702 | -2.4092 | yes | 0.576 |
| 5 | Copula-gaussian | 0.032 | pass | -0.322 | no | 0.663 | -2.4083 | yes | 0.576 |
| 6 | Direct-FHS | 0.024 | pass | +0.020 | yes | 0.776 | -2.4047 | yes | 0.576 |
| 7 | Direct-HS | 0.020 | FAIL | +0.144 | yes | 0.144 | -2.3552 | no | 0.024 |
| 8 | Copula-independence | 0.100 | FAIL | -4.237 | no | 0.000 | -1.2958 | no | 0.000 |

## MCS membership by sub-period (lower alpha)

| model | full_oos | luna | ftx | calm_23 |
|:--|:--:|:--:|:--:|:--:|
| Direct-GJR-GARCH-t | + | + | + | + |
| Direct-GARCH-t | + | + | + | + |
| Direct-FHS | + | + | + | . |
| Copula-clayton | + | + | + | + |
| Copula-student_t | + | + | + | + |
| Copula-gaussian | + | + | + | + |
| Direct-HS | . | + | + | . |
| Copula-independence | . | . | + | + |

## Read

- **Ignoring tail dependence.** `Copula-independence` sits at hit rate 0.065 (target 0.010), ES Z2 -7.41, and is out of the MCS &mdash; treating the 4 assets as independent understates basket tail risk.

- **Dependence structure.** Among the parametric copulas the FZ0 ordering (best first) is clayton < student_t < gaussian; `Copula-clayton` leads with Z2 +0.04.

- **Direct vs copula.** `Direct-GJR-GARCH-t` on the basket return series still edges every copula on FZ0 (-2.2144 vs -2.1808), both refitting volatility daily. Overall FZ0 winner: `Direct-GJR-GARCH-t`.

- The copula marginals are GARCH(1,1)-t; the residual inversion is FHS (empirical). Only the *dependence* is parametric.
- The sub-period MCS has little power (40&ndash;90 day windows), so the copula-family differences show up mainly in the full-sample FZ0 mean.
