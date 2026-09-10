# fit_msgarch_regime_windows.R  --  regime-identification vs. window length.
#
# For the 2-regime MS-GARCH-std, refit walk-forward for several estimation
# windows and record ONLY the filtered probability of the high-variance
# regime (no Risk() call -- much faster than the VaR script). Two specs:
#
#   free  : 2-regime sGARCH(1,1)-std          (same as fit_msgarch_walkforward.R)
#   arch  : 2-regime sARCH(1)-std             (constrained: beta_k = 0, fewer
#           parameters per regime -- the parsimonious variant of V2_PLAN §5.5)
#
#   Rscript msgarch/fit_msgarch_regime_windows.R  [input_csv]  [out_dir]
#
# output: <out_dir>/regime_windows_<ASSET>.csv
#         columns: asset, spec, window, prev_date, date, prob_crisis_filt
#
# WARNING: this is a multi-hour job (6 windows x 2 specs x 2 assets, each an
# MSGARCH walk-forward). Run it deliberately, e.g. overnight.

suppressMessages(library(MSGARCH))

args    <- commandArgs(trailingOnly = TRUE)
IN_CSV  <- if (length(args) >= 1) args[1] else file.path("data", "results", "msgarch_input.csv")
OUT_DIR <- if (length(args) >= 2) args[2] else file.path("data", "results")

WINDOWS     <- list(500, 750, 1000, 1500, 2000, "expanding")
REFIT_EVERY <- 20
OOS_START   <- as.Date("2019-05-16")
MIN_TRAIN   <- 500                      # for the expanding window

SPECS <- list(
  free = CreateSpec(
    variance.spec     = list(model = c("sGARCH", "sGARCH")),
    distribution.spec = list(distribution = c("std", "std")),
    switch.spec       = list(do.mix = FALSE)
  ),
  arch = CreateSpec(
    variance.spec     = list(model = c("sARCH", "sARCH")),
    distribution.spec = list(distribution = c("std", "std")),
    switch.spec       = list(do.mix = FALSE)
  )
)

# crisis = regime with the larger stationary variance (guards label switching)
crisis_k <- function(par) {
  sv <- function(k) {
    a0 <- as.numeric(par[[sprintf("alpha0_%d", k)]])
    a1 <- as.numeric(par[[sprintf("alpha1_%d", k)]])
    bnm <- sprintf("beta_%d", k)
    b  <- if (bnm %in% names(par)) as.numeric(par[[bnm]]) else 0
    d  <- 1 - a1 - b
    if (!is.finite(d) || d <= 0) Inf else a0 / d
  }
  if (sv(2) >= sv(1)) 2L else 1L
}

filt_crisis <- function(fit, newdata, ck) {
  st <- tryCatch(State(object = fit, newdata = newdata), error = function(e) NULL)
  if (is.null(st) || is.null(st$FiltProb)) return(NA_real_)
  as.numeric(st$FiltProb[dim(st$FiltProb)[1], 1, ck])
}

run_one <- function(d, spec, W) {
  d <- d[order(d$date), ]; d <- d[is.finite(d$log_return), ]
  n <- nrow(d)
  expanding <- identical(W, "expanding")
  w0 <- if (expanding) MIN_TRAIN else as.integer(W)
  if (n <= w0 + 5) return(NULL)

  fit <- NULL; ck <- NA_integer_; refit_end <- NA_integer_
  rows <- vector("list", 0)
  for (t in (w0 + 1):n) {
    if (d$date[t] < OOS_START) next
    step <- t - w0
    if (step == 1 || step %% REFIT_EVERY == 1 || is.null(fit)) {
      lo <- if (expanding) 1L else (t - as.integer(W))
      cand <- tryCatch(FitML(spec = spec, data = d$log_return[lo:(t - 1)]),
                       error = function(e) NULL)
      if (!is.null(cand)) { fit <- cand; ck <- crisis_k(fit$par); refit_end <- t - 1 }
    }
    if (is.null(fit)) next
    newdata <- if (!is.na(refit_end) && (t - 1) > refit_end)
      d$log_return[(refit_end + 1):(t - 1)] else numeric(0)
    rows[[length(rows) + 1]] <- data.frame(
      prev_date = d$date[t - 1], date = d$date[t],
      prob_crisis_filt = filt_crisis(fit, newdata, ck)
    )
  }
  if (!length(rows)) return(NULL)
  out <- do.call(rbind, rows)
  out$window <- if (expanding) "expanding" else as.character(W)
  out
}

df <- read.csv(IN_CSV, stringsAsFactors = FALSE)
df$date <- as.Date(df$date)
for (a in unique(df$asset)) {
  da <- df[df$asset == a, ]
  acc <- vector("list", 0)
  for (sn in names(SPECS)) for (W in WINDOWS) {
    cat(sprintf("[%s] spec=%s window=%s\n", a, sn, as.character(W))); flush.console()
    r <- run_one(da, SPECS[[sn]], W)
    if (!is.null(r)) { r$spec <- sn; r$asset <- a; acc[[length(acc) + 1]] <- r }
  }
  res <- do.call(rbind, acc)
  res <- res[, c("asset", "spec", "window", "prev_date", "date", "prob_crisis_filt")]
  fp <- file.path(OUT_DIR, sprintf("regime_windows_%s.csv", a))
  write.csv(res, fp, row.names = FALSE)
  cat(sprintf("  wrote %s (%d rows)\n", fp, nrow(res)))
}
cat("Done.\n")
