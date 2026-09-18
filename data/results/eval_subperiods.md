# Sub-period re-evaluation & conditional predictive ability

_`python -m cryptorisk.study.subperiods`._

## BTC | window 500 | alpha 0.01

MCS membership by period (`+` in set, `.` out):

| model | full_oos | covid | luna | ftx | calm_23 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Realized-SV | + | + | + | + | + |
| Realized-GARCH | + | + | + | + | + |
| CAViaR-SAV | + | + | + | + | + |
| FHS | + | + | + | + | + |
| CAViaR-X-AS | + | + | + | + | + |
| HAR-RV | + | + | + | + | + |
| Jump-Diffusion | + | + | + | + | + |
| LSTM-Vol | + | + | + | + | + |
| HS | + | + | + | + | + |
| RF-QR | + | + | + | + | + |
| GJR-GARCH-t | + | + | + | + | + |
| GARCH-t | + | + | + | . | . |
| CAViaR-AS | + | + | + | + | + |
| MS-GARCH | + | + | + | + | + |
| AWHS | + | + | + | + | + |
| GARCH-EVT | + | + | + | + | . |
| GARCH-X | + | + | + | + | . |
| EWMA | + | + | + | + | + |
| EGARCH-t | + | + | + | + | + |
| HARQ | + | . | + | + | + |

full_oos: 2674d, covid: 71d, luna: 57d, ftx: 61d, calm_23: 92d.

## BTC | window 500 | alpha 0.025

MCS membership by period (`+` in set, `.` out):

| model | full_oos | covid | luna | ftx | calm_23 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Realized-SV | + | + | + | + | + |
| Realized-GARCH | + | + | + | + | + |
| HAR-RV | + | + | + | + | + |
| GJR-GARCH-t | + | + | + | + | + |
| FHS | + | + | + | + | + |
| MS-GARCH | + | + | + | + | + |
| GARCH-t | + | + | + | + | + |
| RF-QR | + | + | + | + | + |
| HS | + | + | + | + | + |
| Jump-Diffusion | + | + | + | + | + |
| CAViaR-AS | + | + | + | + | + |
| CAViaR-X-AS | + | + | + | + | + |
| AWHS | + | + | + | + | + |
| LSTM-Vol | + | + | + | + | + |
| GARCH-EVT | + | + | + | + | . |
| GARCH-X | + | + | + | + | + |
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
| Realized-SV | + | + | + | + | + |
| CAViaR-X-AS | + | + | + | + | + |
| GARCH-X | + | + | + | + | + |
| GARCH-EVT | + | + | + | + | + |
| CAViaR-AS | + | + | + | + | + |
| HAR-RV | + | + | + | + | + |
| EGARCH-t | + | + | + | + | + |
| MS-GARCH | + | + | + | + | + |
| FHS | + | + | + | + | + |
| RF-QR | + | + | + | + | + |
| LSTM-Vol | + | + | + | + | + |
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
| Realized-SV | + | + | + | + | + |
| MS-GARCH | + | + | + | + | + |
| EGARCH-t | + | + | + | + | + |
| FHS | + | + | . | + | + |
| HARQ | + | + | + | + | + |
| CAViaR-AS | + | + | + | + | + |
| RF-QR | + | + | + | + | + |
| AWHS | + | + | + | + | + |
| HS | . | + | + | + | + |
| Jump-Diffusion | . | + | + | + | + |
| EWMA | + | + | + | + | + |
| LSTM-Vol | + | + | + | + | + |
| CAViaR-SAV | + | + | + | + | + |

full_oos: 2674d, covid: 71d, luna: 57d, ftx: 61d, calm_23: 92d.

## Conditional predictive ability (FZ0 loss)

Giacomini-White joint test of *equal conditional* predictive ability (instrument `[1, z(log RV_{t-1})]`), plus a HAC t-test of the RV-state slope of the loss differential `L_best - L_chal`. The joint test is dominated by the unconditional gap (already in DM); the slope is the regime-dependence.

| asset | a | best vs challenger | mean gap | GW p | slope t | slope p | best edge vs RV |
|:--|--:|:--|--:|--:|--:|--:|:--|
| BTC | 0.01 | Realized-SV vs MS-GARCH | -0.1336 | 0.370 | -1.15 | 0.251 | flat |
| BTC | 0.01 | Realized-SV vs HS | -0.1155 | 0.066 | -0.02 | 0.980 | flat |
| BTC | 0.01 | Realized-SV vs EWMA | -0.3966 | 0.046 | -0.75 | 0.452 | flat |
| BTC | 0.01 | Realized-SV vs HARQ | -0.4535 | 0.007 | +0.06 | 0.950 | flat |
| BTC | 0.01 | Realized-SV vs GARCH-t | -0.1227 | 0.004 | +0.36 | 0.716 | flat |
| BTC | 0.01 | Realized-SV vs HAR-RV | -0.0959 | 0.259 | +0.65 | 0.517 | flat |
| BTC | 0.025 | Realized-SV vs MS-GARCH | -0.0619 | 0.391 | -1.11 | 0.267 | flat |
| BTC | 0.025 | Realized-SV vs HS | -0.0732 | 0.071 | -0.10 | 0.918 | flat |
| BTC | 0.025 | Realized-SV vs EWMA | -0.1394 | 0.100 | -0.78 | 0.435 | flat |
| BTC | 0.025 | Realized-SV vs HARQ | -0.1784 | 0.043 | -0.24 | 0.809 | flat |
| BTC | 0.025 | Realized-SV vs GARCH-t | -0.0674 | 0.039 | -0.43 | 0.668 | flat |
| BTC | 0.025 | Realized-SV vs HAR-RV | -0.0133 | 0.542 | +0.82 | 0.415 | flat |
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
