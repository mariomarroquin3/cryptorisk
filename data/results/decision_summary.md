# Decision layer

_`python -m cryptorisk.study.run_decision`._

Notional $1,000,000 | liquidity horizon 10d | Basel base multiplier 1.5 | 1-day 99% ES budget $120,000.

## Capital (FRTB ES-IMA, 97.5% ES, MCS models)

| asset | model | m_c | ES 10d √t | ES 10d boot | capital $ |
|:--|:--|--:|--:|--:|--:|
| BTC | HAR-RV | 1.50 | -0.2347 | -0.2904 | 352,057 |
| BTC | MS-GARCH | 1.50 | -0.2750 | -0.2904 | 412,569 |
| BTC | CAViaR-X-AS | 1.50 | -0.2808 | -0.2904 | 421,255 |
| BTC | EWMA | 1.90 | -0.2219 | -0.2904 | 421,629 |
| BTC | HARQ | 1.90 | -0.2264 | -0.2904 | 430,177 |
| BTC | AWHS | 1.50 | -0.2873 | -0.2904 | 430,936 |
| BTC | FHS | 1.50 | -0.2882 | -0.2904 | 432,358 |
| BTC | Jump-Diffusion | 1.50 | -0.2928 | -0.2904 | 439,183 |
| BTC | Realized-GARCH | 1.50 | -0.2954 | -0.2904 | 443,045 |
| BTC | HS | 1.50 | -0.2971 | -0.2904 | 445,654 |
| BTC | GJR-GARCH-t | 1.50 | -0.3060 | -0.2904 | 458,928 |
| BTC | GARCH-t | 1.50 | -0.3065 | -0.2904 | 459,751 |
| BTC | EGARCH-t | 1.50 | -0.3461 | -0.2904 | 519,209 |
| BTC | CAViaR-AS | 1.90 | -0.2820 | -0.2904 | 535,728 |
| BTC | CAViaR-SAV | 1.90 | -0.2848 | -0.2904 | 541,048 |
| BTC | GARCH-X | 1.50 | -0.3665 | -0.2904 | 549,688 |
| BTC | GARCH-EVT | 1.90 | -0.3053 | -0.2904 | 580,009 |
| ETH | HARQ | 1.50 | -0.2908 | -0.3855 | 436,260 |
| ETH | MS-GARCH | 1.50 | -0.3559 | -0.3855 | 533,910 |
| ETH | EWMA | 1.90 | -0.2920 | -0.3855 | 554,878 |
| ETH | CAViaR-X-AS | 1.50 | -0.3737 | -0.3855 | 560,597 |
| ETH | CAViaR-AS | 1.50 | -0.3742 | -0.3855 | 561,264 |
| ETH | HAR-RV | 1.90 | -0.2986 | -0.3855 | 567,305 |
| ETH | Realized-GARCH | 1.50 | -0.3783 | -0.3855 | 567,477 |
| ETH | FHS | 1.50 | -0.3806 | -0.3855 | 570,918 |
| ETH | GARCH-EVT | 1.50 | -0.3833 | -0.3855 | 574,889 |
| ETH | CAViaR-SAV | 1.50 | -0.3866 | -0.3855 | 579,856 |
| ETH | AWHS | 1.50 | -0.3880 | -0.3855 | 581,974 |
| ETH | GARCH-t | 1.50 | -0.3927 | -0.3855 | 589,056 |
| ETH | GJR-GARCH-t | 1.50 | -0.3941 | -0.3855 | 591,081 |
| ETH | GARCH-X | 1.50 | -0.4121 | -0.3855 | 618,146 |
| ETH | EGARCH-t | 1.50 | -0.4144 | -0.3855 | 621,567 |

BTC: model-risk add-on (capital spread across the MCS) = $227,953.


ETH: model-risk add-on (capital spread across the MCS) = $185,306.

## Estimation-risk band (final estimation window)

Parameter / sampling uncertainty on the last 500-day window, three archetypes: HS (stationary block bootstrap), GARCH-t (draw from the fitted asymptotic covariance, re-forecast), FHS (parameter draw + residual resample). `ES 97.5% p5` is the prudent (conservative) draw; the add-on is `capital(prudent ES) - capital(point ES)` at the Basel base multiplier, so it isolates estimation risk.

| asset | estimator | ES 97.5% point | ES s.e. | ES 97.5% p5 (prudent) | abs-ES widening | est.-risk add-on $ |
|:--|:--|--:|--:|--:|--:|--:|
| BTC | FHS | -0.0480 | 0.0062 | -0.0584 | +0.0104 | 49,370 |
| BTC | GARCH-t | -0.0533 | 0.0048 | -0.0600 | +0.0068 | 32,157 |
| BTC | HS | -0.0595 | 0.0090 | -0.0735 | +0.0140 | 66,396 |
| ETH | FHS | -0.0807 | 0.0115 | -0.0961 | +0.0154 | 72,933 |
| ETH | GARCH-t | -0.0884 | 0.0132 | -0.1017 | +0.0133 | 62,913 |
| ETH | HS | -0.0913 | 0.0100 | -0.1053 | +0.0140 | 66,183 |

## Position limit N* (mean 1-day 99% ES = budget) + framework backtest

`mean util` is 1.00 by construction (N* set so the *average* ES = budget); `max util` and the breach rates are the informative columns.

| asset | model | N* $ | bind rate | max util | budget breach | ES exceed (~1%) | worst loss $ |
|:--|:--|--:|--:|--:|--:|--:|--:|
| BTC | GARCH-X | 763,906 | 0.297 | 10.07 | 0.0011 | 0.0034 | 359,466 |
| BTC | EGARCH-t | 786,973 | 0.347 | 16.73 | 0.0011 | 0.0030 | 370,320 |
| BTC | GARCH-t | 895,601 | 0.407 | 5.20 | 0.0026 | 0.0030 | 421,437 |
| BTC | GJR-GARCH-t | 899,846 | 0.408 | 5.61 | 0.0026 | 0.0030 | 423,434 |
| BTC | GARCH-EVT | 927,391 | 0.371 | 6.80 | 0.0026 | 0.0045 | 436,396 |
| BTC | Realized-GARCH | 949,269 | 0.398 | 5.25 | 0.0030 | 0.0022 | 446,691 |
| BTC | HS | 985,070 | 0.484 | 1.69 | 0.0030 | 0.0026 | 463,538 |
| BTC | Jump-Diffusion | 1,000,905 | 0.593 | 1.75 | 0.0030 | 0.0037 | 470,989 |
| BTC | FHS | 1,021,097 | 0.405 | 8.89 | 0.0034 | 0.0041 | 480,491 |
| BTC | AWHS | 1,040,083 | 0.513 | 2.64 | 0.0041 | 0.0052 | 489,425 |
| BTC | CAViaR-SAV | 1,075,111 | 0.480 | 5.77 | 0.0045 | 0.0049 | 505,907 |
| BTC | CAViaR-AS | 1,088,694 | 0.468 | 4.59 | 0.0049 | 0.0060 | 512,299 |
| BTC | CAViaR-X-AS | 1,102,366 | 0.454 | 4.38 | 0.0049 | 0.0052 | 518,733 |
| BTC | MS-GARCH | 1,120,608 | 0.487 | 3.49 | 0.0049 | 0.0037 | 527,316 |
| BTC | HAR-RV | 1,406,302 | 0.421 | 4.45 | 0.0123 | 0.0079 | 661,754 |
| BTC | HARQ | 1,457,809 | 0.387 | 5.37 | 0.0135 | 0.0150 | 685,991 |
| BTC | EWMA | 1,499,960 | 0.414 | 3.99 | 0.0142 | 0.0146 | 705,826 |
| ETH | EGARCH-t | 676,818 | 0.345 | 11.53 | 0.0022 | 0.0037 | 382,818 |
| ETH | GARCH-X | 686,656 | 0.371 | 8.57 | 0.0022 | 0.0030 | 388,382 |
| ETH | GJR-GARCH-t | 711,250 | 0.396 | 5.86 | 0.0022 | 0.0037 | 402,293 |
| ETH | GARCH-t | 712,339 | 0.394 | 6.26 | 0.0022 | 0.0030 | 402,909 |
| ETH | HS | 722,694 | 0.588 | 1.64 | 0.0022 | 0.0045 | 408,766 |
| ETH | Jump-Diffusion | 724,453 | 0.591 | 1.74 | 0.0022 | 0.0052 | 409,761 |
| ETH | Realized-GARCH | 748,858 | 0.359 | 6.81 | 0.0037 | 0.0030 | 423,565 |
| ETH | AWHS | 767,016 | 0.509 | 2.10 | 0.0041 | 0.0056 | 433,835 |
| ETH | FHS | 773,430 | 0.422 | 7.02 | 0.0045 | 0.0052 | 437,463 |
| ETH | GARCH-EVT | 775,502 | 0.410 | 6.56 | 0.0045 | 0.0049 | 438,635 |
| ETH | CAViaR-SAV | 776,454 | 0.426 | 4.61 | 0.0045 | 0.0064 | 439,173 |
| ETH | CAViaR-AS | 777,809 | 0.398 | 5.49 | 0.0045 | 0.0045 | 439,940 |
| ETH | CAViaR-X-AS | 784,559 | 0.397 | 5.47 | 0.0052 | 0.0045 | 443,758 |
| ETH | MS-GARCH | 862,531 | 0.448 | 3.35 | 0.0071 | 0.0056 | 487,860 |
| ETH | HAR-RV | 1,105,665 | 0.406 | 4.73 | 0.0138 | 0.0097 | 625,380 |
| ETH | HARQ | 1,135,093 | 0.396 | 4.86 | 0.0142 | 0.0150 | 642,025 |
| ETH | EWMA | 1,139,758 | 0.418 | 3.84 | 0.0142 | 0.0153 | 644,664 |

## FRTB PLA test (RTPL vs HPL)

RTPL is the realized outcome mapped through the model's predictive CDF, so Spearman is ~1 by construction; the KS distance (predictive *shape* vs realized) is what separates the models.

| asset | model | Spearman | KS | zone |
|:--|:--|--:|--:|:--:|
| BTC | EWMA | 1.000 | 0.001 | green |
| BTC | HARQ | 1.000 | 0.006 | green |
| BTC | HAR-RV | 1.000 | 0.006 | green |
| BTC | Jump-Diffusion | 0.998 | 0.031 | green |
| BTC | Realized-GARCH | 0.998 | 0.064 | green |
| BTC | GARCH-X | 0.998 | 0.064 | green |
| BTC | GJR-GARCH-t | 0.995 | 0.085 | green |
| BTC | GARCH-t | 0.995 | 0.089 | green |
| BTC | FHS | 0.996 | 0.090 | green |
| BTC | EGARCH-t | 0.987 | 0.096 | amber |
| BTC | AWHS | 0.993 | 0.096 | amber |
| BTC | HS | 0.995 | 0.097 | amber |
| BTC | GARCH-EVT | 0.994 | 0.100 | amber |
| ETH | EWMA | 1.000 | 0.001 | green |
| ETH | HAR-RV | 1.000 | 0.006 | green |
| ETH | HARQ | 1.000 | 0.006 | green |
| ETH | GARCH-X | 0.998 | 0.064 | green |
| ETH | FHS | 0.995 | 0.064 | green |
| ETH | Realized-GARCH | 0.998 | 0.064 | green |
| ETH | GJR-GARCH-t | 0.995 | 0.074 | green |
| ETH | EGARCH-t | 0.990 | 0.074 | green |
| ETH | GARCH-t | 0.995 | 0.075 | green |
| ETH | GARCH-EVT | 0.994 | 0.076 | green |
| ETH | AWHS | 0.993 | 0.078 | green |

## Perpetual hedge (perp return proxied by spot)

`funding carry` is the annualised funding on the hedged notional; **positive = the short-perp hedge earns it** (longs pay shorts).

| asset | h (min-var) | h (ES-min) | ES unhedged | ES hedged | ES reduction | funding carry $/yr |
|:--|--:|--:|--:|--:|--:|--:|
| BTC | 1.000 | n/a | -0.0962 | n/a | n/a | +115,834 |
| ETH | 1.000 | n/a | -0.1269 | n/a | n/a | +138,386 |

_perp return proxied by spot (no perp price in store); ES-min hedge and ES reduction not meaningful, basis risk unavailable._
