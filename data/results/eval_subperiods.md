# Sub-period re-evaluation & conditional predictive ability

_`python -m cryptorisk.study.subperiods`._

## BTC | window 500 | alpha 0.01

MCS membership by period (`+` in set, `.` out):

| model | full_oos | covid | luna | ftx | calm_23 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Realized-GARCH | + | + | + | + | + |
| CAViaR-SAV | + | + | + | + | + |
| FHS | + | + | + | + | + |
| CAViaR-X-AS | + | + | + | + | + |
| HAR-RV | + | + | + | + | + |
| Jump-Diffusion | + | + | + | + | + |
| HS | + | + | + | + | + |
| GJR-GARCH-t | + | + | + | + | + |
| GARCH-t | + | + | + | + | + |
| CAViaR-AS | + | + | + | + | + |
| MS-GARCH | + | + | + | + | + |
| AWHS | + | + | + | + | + |
| GARCH-EVT | + | + | + | + | + |
| GARCH-X | + | + | + | + | . |
| EWMA | + | + | + | + | + |
| EGARCH-t | + | + | + | + | + |
| HARQ | + | . | + | + | + |

full_oos: 2674d, covid: 71d, luna: 57d, ftx: 61d, calm_23: 92d.

## BTC | window 500 | alpha 0.025

MCS membership by period (`+` in set, `.` out):

| model | full_oos | covid | luna | ftx | calm_23 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Realized-GARCH | + | + | + | + | + |
| HAR-RV | + | + | + | + | + |
| GJR-GARCH-t | + | + | + | + | + |
| FHS | + | + | + | + | + |
| MS-GARCH | + | + | + | + | + |
| GARCH-t | + | + | + | + | + |
| HS | + | + | + | + | + |
| Jump-Diffusion | + | + | + | + | + |
| CAViaR-AS | + | + | + | + | + |
| CAViaR-X-AS | + | + | + | + | + |
| AWHS | + | + | + | + | + |
| GARCH-EVT | + | + | + | + | . |
| GARCH-X | + | + | + | + | . |
| CAViaR-SAV | + | + | + | + | + |
| EWMA | + | + | + | + | + |
| HARQ | + | . | + | + | + |
| EGARCH-t | + | + | + | + | + |

full_oos: 2674d, covid: 71d, luna: 57d, ftx: 61d, calm_23: 92d.

## ETH | window 500 | alpha 0.01

MCS membership by period (`+` in set, `.` out):

| model | full_oos | covid | luna | ftx | calm_23 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Realized-GARCH | + | + | + | + | + |
| GARCH-t | + | + | + | + | + |
| GJR-GARCH-t | + | + | + | + | + |
| CAViaR-X-AS | + | + | + | + | + |
| GARCH-X | + | + | + | + | + |
| GARCH-EVT | + | + | + | + | + |
| CAViaR-AS | + | + | + | + | + |
| HAR-RV | + | + | + | + | + |
| EGARCH-t | + | + | + | + | + |
| MS-GARCH | + | + | + | + | + |
| FHS | + | + | + | + | + |
| Jump-Diffusion | + | + | + | + | + |
| AWHS | + | + | + | + | + |
| HS | + | + | + | + | + |
| HARQ | + | + | + | + | + |
| CAViaR-SAV | + | + | + | + | + |
| EWMA | + | + | + | + | + |

full_oos: 2674d, covid: 71d, luna: 57d, ftx: 61d, calm_23: 92d.

## ETH | window 500 | alpha 0.025

MCS membership by period (`+` in set, `.` out):

| model | full_oos | covid | luna | ftx | calm_23 |
|:--|:--:|:--:|:--:|:--:|:--:|
| HAR-RV | + | + | + | + | + |
| Realized-GARCH | + | + | + | + | + |
| GARCH-t | + | + | + | + | + |
| GJR-GARCH-t | + | + | + | + | + |
| GARCH-X | + | + | + | + | + |
| CAViaR-X-AS | + | + | + | + | + |
| GARCH-EVT | + | + | + | + | + |
| MS-GARCH | + | + | + | + | + |
| EGARCH-t | + | + | + | + | + |
| FHS | + | + | . | + | + |
| HARQ | + | + | + | + | + |
| CAViaR-AS | + | + | + | + | + |
| AWHS | + | + | + | + | + |
| HS | . | + | + | + | + |
| Jump-Diffusion | . | + | + | + | + |
| EWMA | + | + | + | + | + |
| CAViaR-SAV | + | + | + | + | + |

full_oos: 2674d, covid: 71d, luna: 57d, ftx: 61d, calm_23: 92d.

## Conditional predictive ability (FZ0 loss)

Giacomini-White joint test of *equal conditional* predictive ability (instrument `[1, z(log RV_{t-1})]`), plus a HAC t-test of the RV-state slope of the loss differential `L_best - L_chal`. The joint test is dominated by the unconditional gap (already in DM); the slope is the regime-dependence.

| asset | a | best vs challenger | mean gap | GW p | slope t | slope p | best edge vs RV |
|:--|--:|:--|--:|--:|--:|--:|:--|
| BTC | 0.01 | Realized-GARCH vs MS-GARCH | -0.0916 | 0.520 | -1.00 | 0.316 | flat |
| BTC | 0.01 | Realized-GARCH vs HS | -0.0736 | 0.261 | +0.51 | 0.608 | flat |
| BTC | 0.01 | Realized-GARCH vs EWMA | -0.3546 | 0.029 | -0.64 | 0.523 | flat |
| BTC | 0.01 | Realized-GARCH vs HARQ | -0.4115 | 0.005 | +0.29 | 0.770 | flat |
| BTC | 0.01 | Realized-GARCH vs GARCH-t | -0.0807 | 0.009 | +1.39 | 0.165 | flat |
| BTC | 0.01 | Realized-GARCH vs HAR-RV | -0.0539 | 0.068 | +1.36 | 0.173 | flat |
| BTC | 0.025 | Realized-GARCH vs MS-GARCH | -0.0540 | 0.401 | -0.83 | 0.405 | flat |
| BTC | 0.025 | Realized-GARCH vs HS | -0.0653 | 0.131 | +0.39 | 0.697 | flat |
| BTC | 0.025 | Realized-GARCH vs EWMA | -0.1315 | 0.042 | -0.48 | 0.632 | flat |
| BTC | 0.025 | Realized-GARCH vs HARQ | -0.1705 | 0.028 | +0.08 | 0.938 | flat |
| BTC | 0.025 | Realized-GARCH vs GARCH-t | -0.0595 | 0.028 | +0.34 | 0.737 | flat |
| BTC | 0.025 | Realized-GARCH vs HAR-RV | -0.0054 | 0.152 | +1.63 | 0.103 | flat |
| ETH | 0.01 | Realized-GARCH vs MS-GARCH | -0.0508 | 0.369 | -1.65 | 0.099 | flat |
| ETH | 0.01 | Realized-GARCH vs HS | -0.1564 | 0.012 | +0.02 | 0.984 | flat |
| ETH | 0.01 | Realized-GARCH vs EWMA | -0.2889 | 0.015 | -1.72 | 0.086 | flat |
| ETH | 0.01 | Realized-GARCH vs HARQ | -0.1717 | 0.065 | +1.41 | 0.157 | flat |
| ETH | 0.01 | Realized-GARCH vs GARCH-t | -0.0177 | 0.568 | +0.41 | 0.682 | flat |
| ETH | 0.01 | Realized-GARCH vs HAR-RV | -0.0444 | 0.706 | -0.10 | 0.921 | flat |
| ETH | 0.025 | HAR-RV vs MS-GARCH | -0.0351 | 0.093 | -2.17 | 0.030 | grows in high RV |
| ETH | 0.025 | HAR-RV vs HS | -0.1148 | 0.008 | -0.73 | 0.463 | flat |
| ETH | 0.025 | HAR-RV vs EWMA | -0.1215 | 0.001 | -1.99 | 0.046 | grows in high RV |
| ETH | 0.025 | HAR-RV vs HARQ | -0.0631 | 0.121 | +1.55 | 0.121 | flat |
| ETH | 0.025 | HAR-RV vs GARCH-t | -0.0260 | 0.542 | -0.84 | 0.399 | flat |
