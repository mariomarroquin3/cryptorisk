# MS-GARCH regime identification vs. estimation window

_`python -m cryptorisk.study.regime_identification`._

Correlation of the high-variance-regime probability with the realized volatility state.

| asset | series | corr(\|r\|) | Spearman(\|r\|) | corr(RV) | corr(RV21) |
|:--|:--|--:|--:|--:|--:|
| BTC | filt_wf (walk-forward W=500) | 0.014 | 0.030 | 0.005 | -0.045 |
| BTC | pred_wf (walk-forward W=500) | -0.057 | -0.035 | -0.061 | -0.098 |
| BTC | insample (full-sample fit) | 0.698 | 0.913 | 0.151 | 0.039 |
| ETH | filt_wf (walk-forward W=500) | -0.034 | -0.026 | -0.024 | -0.131 |
| ETH | pred_wf (walk-forward W=500) | -0.060 | -0.054 | -0.062 | -0.177 |
| ETH | insample (full-sample fit) | 0.555 | 0.817 | 0.051 | 0.015 |

Read: the full-sample fit tracks the volatility state; the walk-forward W=500 filtered/predicted probability barely does. The regime layer is therefore descriptive only (labelled in-sample) and is kept out of the VaR backtest.
