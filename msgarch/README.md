# MS-GARCH bridge (R)

MS-GARCH is estimated in R via the `MSGARCH` 2.51 package and consumed from
Python through `cryptorisk.models.msgarch_bridge` (subprocess + CSV, the pattern
proven in v1). It is **not** reimplemented in Python.

Phase 2: port `fit_msgarch_walkforward.R` from v1, adapting it to the v2 store
(read `returns_daily`, write results as tidy CSV keyed by `asset`).

Key v1 facts to carry over:
- API 2.51: `FitML`, `Risk`, `State`, `Volatility`, `TransMat` exist; `Smooth`,
  `FilterML` and the `coef` method do **not** (params are in `fit$par`).
- `Risk()` / `State()` accept `newdata=` which is appended and re-filters.
- Regime labelling by stationary variance each refit (avoids label switching).
- Record `State()$FiltProb` (endpoint `SmoothProb` degenerates to the ergodic
  mean); also export a single full-sample fit's `FiltProb` as the descriptive
  `prob_high_vol_insample` (walk-forward regime prob has ~no OOS signal - see
  V2_PLAN §5.5).
