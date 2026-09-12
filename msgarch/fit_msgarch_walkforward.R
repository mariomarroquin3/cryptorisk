# fit_msgarch_walkforward.R  --  MS-GARCH(1,1)-std, 2 regimes, walk-forward.
#
# Ported from the v1 project (see msgarch/README.md). Runs entirely in R because
# MSGARCH's walk-forward is not day-by-day through the Python engine; the results
# are cached in the store and served by cryptorisk.models.msgarch_bridge.
#
# MSGARCH 2.51 API used: FitML, Risk(alpha=, nahead=1, newdata=), State()$FiltProb
# / $PredProb, TransMat, predict(nahead=1)$vol. Smooth() / FilterML() and the
# coef method do NOT exist.
#
#   Rscript msgarch/fit_msgarch_walkforward.R  [input_csv]  [out_dir]
#
# input_csv columns: asset, date, log_return   (default data/results/msgarch_input.csv)
# output: <out_dir>/msgarch_pred_<ASSET>.csv per asset.

suppressMessages(library(MSGARCH))

args    <- commandArgs(trailingOnly = TRUE)
IN_CSV  <- if (length(args) >= 1) args[1] else file.path("data", "results", "msgarch_input.csv")
OUT_DIR <- if (length(args) >= 2) args[2] else file.path("data", "results")

WINDOW      <- 500
REFIT_EVERY <- 20
ALPHAS      <- c(0.025, 0.01)
OOS_START   <- as.Date("2019-05-16")   # matches config/study.yaml sample.oos_start

spec <- CreateSpec(
  variance.spec     = list(model = c("sGARCH", "sGARCH")),
  distribution.spec = list(distribution = c("std", "std")),
  switch.spec       = list(do.mix = FALSE)
)

# crisis = regime with the larger stationary variance (guards label switching)
crisis_k <- function(par) {
  sv <- function(k) {
    a0 <- as.numeric(par[[sprintf("alpha0_%d", k)]])
    a1 <- as.numeric(par[[sprintf("alpha1_%d", k)]])
    b  <- as.numeric(par[[sprintf("beta_%d",   k)]])
    d  <- 1 - a1 - b
    if (!is.finite(d) || d <= 0) Inf else a0 / d
  }
  if (sv(2) >= sv(1)) 2L else 1L
}

prob_crisis <- function(fit, newdata, ck) {
  st <- tryCatch(State(object = fit, newdata = newdata), error = function(e) NULL)
  if (is.null(st)) return(c(NA_real_, NA_real_))
  gv <- function(a) if (is.null(a)) NA_real_ else as.numeric(a[dim(a)[1], 1, ck])
  c(gv(st$FiltProb), gv(st$PredProb))
}

run_asset <- function(d) {
  d <- d[order(d$date), ]
  d <- d[is.finite(d$log_return), ]
  n <- nrow(d)
  if (n <= WINDOW + 5) stop(sprintf("insufficient rows (%d)", n))

  fit <- NULL; ck <- NA_integer_; refit_end <- NA_integer_
  rows <- vector("list", 0)

  for (t in (WINDOW + 1):n) {
    if (d$date[t] < OOS_START) next
    step <- t - WINDOW
    if (step == 1 || step %% REFIT_EVERY == 1 || is.null(fit)) {
      w <- d$log_return[(t - WINDOW):(t - 1)]
      cand <- tryCatch(FitML(spec = spec, data = w), error = function(e) NULL)
      if (!is.null(cand)) { fit <- cand; ck <- crisis_k(fit$par); refit_end <- t - 1 }
    }
    if (is.null(fit)) next

    newdata <- if (!is.na(refit_end) && (t - 1) > refit_end)
      d$log_return[(refit_end + 1):(t - 1)] else numeric(0)

    rk <- tryCatch(Risk(fit, alpha = ALPHAS, nahead = 1, newdata = newdata),
                   error = function(e) NULL)
    if (is.null(rk)) next
    vol <- tryCatch(as.numeric(predict(fit, nahead = 1, newdata = newdata)$vol[1]),
                    error = function(e) NA_real_)
    pc <- prob_crisis(fit, newdata, ck)

    rows[[length(rows) + 1]] <- data.frame(
      prev_date = d$date[t - 1], date = d$date[t],
      var_0025 = rk$VaR[1, 1], es_0025 = rk$ES[1, 1],
      var_001  = rk$VaR[1, 2], es_001  = rk$ES[1, 2],
      sigma2 = vol^2,
      prob_crisis_filt = pc[1], prob_crisis_pred = pc[2]
    )
  }
  out <- do.call(rbind, rows)

  # descriptive in-sample regime curve (single full-sample fit; NOT in the VaR
  # backtest -- see V2_PLAN §5.5)
  ff <- tryCatch(FitML(spec = spec, data = d$log_return), error = function(e) NULL)
  ins <- data.frame(date = d$date, prob_crisis_insample = NA_real_)
  regimes <- NULL
  if (!is.null(ff)) {
    st <- tryCatch(State(ff), error = function(e) NULL)
    if (!is.null(st$FiltProb)) {
      k <- crisis_k(ff$par)
      ins$prob_crisis_insample <- as.numeric(st$FiltProb[seq_len(nrow(d)), 1, k])
    }
    regimes <- regime_params(ff, crisis_k(ff$par))
  }
  list(preds = merge(out, ins, by = "date", all.x = TRUE), regimes = regimes)
}

# Per-regime GARCH params from the full-sample fit `ff` (descriptive only, same
# fit used for `prob_crisis_insample`) -- for a labeled "if a crisis regime
# persists" scenario band, NOT a walk-forward forecast (see the module
# docstring's OOS-weakness caveat). `stat_vol` is the regime-conditional
# stationary vol sqrt(alpha0/(1-alpha1-beta)) (same formula as `crisis_k`'s
# labeling rule); `p_stay` is the regime's self-transition probability
# (TransMat diagonal) -- how persistent that regime is once entered.
regime_params <- function(fit, ck) {
  stat_vol <- function(k) {
    a0 <- as.numeric(fit$par[[sprintf("alpha0_%d", k)]])
    a1 <- as.numeric(fit$par[[sprintf("alpha1_%d", k)]])
    b  <- as.numeric(fit$par[[sprintf("beta_%d",   k)]])
    d  <- 1 - a1 - b
    if (!is.finite(d) || d <= 0) NA_real_ else sqrt(a0 / d)
  }
  tm <- tryCatch(TransMat(fit), error = function(e) NULL)
  p_stay <- function(k) if (is.null(tm)) NA_real_ else as.numeric(tm[k, k])
  filt_vol_end <- tryCatch(tail(as.numeric(Volatility(fit)), 1), error = function(e) NA_real_)
  data.frame(
    regime = c(if (ck == 1) "crisis" else "normal", if (ck == 2) "crisis" else "normal"),
    alpha0 = c(fit$par[["alpha0_1"]], fit$par[["alpha0_2"]]),
    alpha1 = c(fit$par[["alpha1_1"]], fit$par[["alpha1_2"]]),
    beta   = c(fit$par[["beta_1"]],   fit$par[["beta_2"]]),
    nu     = c(fit$par[["nu_1"]],     fit$par[["nu_2"]]),
    stat_vol = c(stat_vol(1), stat_vol(2)),
    p_stay   = c(p_stay(1), p_stay(2)),
    filt_vol_end = filt_vol_end
  )
}

df <- read.csv(IN_CSV, stringsAsFactors = FALSE)
df$date <- as.Date(df$date)
all_regimes <- vector("list", 0)
for (a in unique(df$asset)) {
  cat(sprintf("MS-GARCH walk-forward: %s\n", a))
  out <- run_asset(df[df$asset == a, ])
  res <- out$preds
  res$asset <- a
  res <- res[order(res$date), c("asset", "prev_date", "date", "var_0025", "es_0025",
                                "var_001", "es_001", "sigma2",
                                "prob_crisis_filt", "prob_crisis_pred",
                                "prob_crisis_insample")]
  fp <- file.path(OUT_DIR, sprintf("msgarch_pred_%s.csv", a))
  write.csv(res, fp, row.names = FALSE)
  cat(sprintf("  wrote %s (%d rows, %s -> %s)\n", fp, nrow(res),
              min(res$date), max(res$date)))
  if (!is.null(out$regimes)) {
    out$regimes$asset <- a
    all_regimes[[length(all_regimes) + 1]] <- out$regimes
  }
}
if (length(all_regimes) > 0) {
  rp <- do.call(rbind, all_regimes)
  rp <- rp[, c("asset", "regime", "alpha0", "alpha1", "beta", "nu",
               "stat_vol", "p_stay", "filt_vol_end")]
  fp <- file.path(OUT_DIR, "msgarch_regime_params.csv")
  write.csv(rp, fp, row.names = FALSE)
  cat(sprintf("  wrote %s (%d rows)\n", fp, nrow(rp)))
}
cat("Done.\n")
