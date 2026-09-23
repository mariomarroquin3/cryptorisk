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
| LSTM-Vol | + | + | + | + | + |
| Jump-Diffusion | + | + | + | + | + |
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
| LSTM-Vol | + | + | + | + | + |
| AWHS | + | + | + | + | + |
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
| LSTM-Vol | + | + | + | + | + |
| RF-QR | + | + | + | + | + |
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
| LSTM-Vol | + | + | + | + | + |
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
| BTC | 0.01 | Realized-SV vs MS-GARCH | -0.1338 | 0.368 | -1.15 | 0.252 | flat |
| BTC | 0.01 | Realized-SV vs HS | -0.1158 | 0.064 | -0.02 | 0.986 | flat |
| BTC | 0.01 | Realized-SV vs EWMA | -0.3968 | 0.045 | -0.76 | 0.449 | flat |
| BTC | 0.01 | Realized-SV vs HARQ | -0.4535 | 0.007 | +0.05 | 0.961 | flat |
| BTC | 0.01 | Realized-SV vs GARCH-t | -0.1230 | 0.004 | +0.37 | 0.709 | flat |
| BTC | 0.01 | Realized-SV vs HAR-RV | -0.0960 | 0.260 | +0.64 | 0.523 | flat |
| BTC | 0.025 | Realized-SV vs MS-GARCH | -0.0621 | 0.388 | -1.11 | 0.269 | flat |
| BTC | 0.025 | Realized-SV vs HS | -0.0735 | 0.069 | -0.10 | 0.923 | flat |
| BTC | 0.025 | Realized-SV vs EWMA | -0.1397 | 0.099 | -0.78 | 0.438 | flat |
| BTC | 0.025 | Realized-SV vs HARQ | -0.1783 | 0.043 | -0.26 | 0.798 | flat |
| BTC | 0.025 | Realized-SV vs GARCH-t | -0.0677 | 0.038 | -0.41 | 0.680 | flat |
| BTC | 0.025 | Realized-SV vs HAR-RV | -0.0134 | 0.544 | +0.81 | 0.420 | flat |
| ETH | 0.01 | Realized-GARCH vs MS-GARCH | -0.0508 | 0.370 | -1.65 | 0.100 | flat |
| ETH | 0.01 | Realized-GARCH vs HS | -0.1564 | 0.011 | +0.02 | 0.985 | flat |
| ETH | 0.01 | Realized-GARCH vs EWMA | -0.2890 | 0.015 | -1.73 | 0.084 | flat |
| ETH | 0.01 | Realized-GARCH vs HARQ | -0.1718 | 0.065 | +1.40 | 0.163 | flat |
| ETH | 0.01 | Realized-GARCH vs GARCH-t | -0.0178 | 0.562 | +0.41 | 0.678 | flat |
| ETH | 0.01 | Realized-GARCH vs HAR-RV | -0.0445 | 0.708 | -0.11 | 0.910 | flat |
| ETH | 0.025 | HAR-RV vs MS-GARCH | -0.0351 | 0.096 | -2.15 | 0.032 | grows in high RV |
| ETH | 0.025 | HAR-RV vs HS | -0.1148 | 0.008 | -0.72 | 0.469 | flat |
| ETH | 0.025 | HAR-RV vs EWMA | -0.1215 | 0.001 | -1.99 | 0.047 | grows in high RV |
| ETH | 0.025 | HAR-RV vs HARQ | -0.0631 | 0.121 | +1.54 | 0.123 | flat |
| ETH | 0.025 | HAR-RV vs GARCH-t | -0.0259 | 0.549 | -0.83 | 0.407 | flat |
