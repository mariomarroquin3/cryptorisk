# bootstrap_pcrisis.R -- parameter-uncertainty band for the walk-forward
# P(crisis).
#
# At every STEP-th out-of-sample day, take the same 500-day estimation window
# the walk-forward uses, draw B stationary-block bootstrap resamples of it
# (mean block length BLOCK), refit the 2-regime MS-GARCH on each resample, then
# run each refit's parameters over the ORIGINAL window and record the filtered
# P(crisis) at its last day. The spread across resamples is how much of the
# regime probability is decided by estimation noise rather than by the data:
# a wide band means the regimes are not identified in a 500-day window.
#
#   Rscript msgarch/bootstrap_pcrisis.R [B] [STEP] [input_csv] [out_csv] [asset]
#
# The optional asset restricts the run to one ticker so the two assets can run
# in parallel processes (each writes its own out_csv).
#
# output columns: asset, date, p_point, p_lo, p_med, p_hi, p_sd, n_ok

suppressMessages(library(MSGARCH))

args    <- commandArgs(trailingOnly = TRUE)
B       <- if (length(args) >= 1) as.integer(args[1]) else 20L
STEP    <- if (length(args) >= 2) as.integer(args[2]) else 60L
IN_CSV  <- if (length(args) >= 3) args[3] else file.path("data", "results", "msgarch_input.csv")
OUT_CSV <- if (length(args) >= 4) args[4] else file.path("data", "results", "msgarch_pcrisis_band.csv")
ONLY    <- if (length(args) >= 5) args[5] else NA_character_

WINDOW    <- 500
BLOCK     <- 20
OOS_START <- as.Date("2019-05-16")
SEED      <- 20260101

spec <- CreateSpec(
  variance.spec     = list(model = c("sGARCH", "sGARCH")),
  distribution.spec = list(distribution = c("std", "std")),
  switch.spec       = list(do.mix = FALSE)
)

# crisis = regime with the larger stationary variance (same rule as the
# walk-forward script, guards label switching across resamples)
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

# P(crisis) at the last day of `orig`, using `fit`'s parameters
p_end <- function(fit, orig) {
  st <- tryCatch(State(object = fit, newdata = orig), error = function(e) NULL)
  if (is.null(st) || is.null(st$FiltProb)) return(NA_real_)
  a <- st$FiltProb
  as.numeric(a[dim(a)[1], 1, crisis_k(fit$par)])
}

# stationary block bootstrap (Politis-Romano): geometric block lengths
block_boot <- function(x, mean_block) {
  n <- length(x); out <- numeric(n); i <- 1L
  while (i <= n) {
    start <- sample.int(n, 1L); len <- rgeom(1L, 1 / mean_block) + 1L
    idx <- ((start - 1L + seq_len(len) - 1L) %% n) + 1L
    take <- min(len, n - i + 1L)
    out[i:(i + take - 1L)] <- x[idx[seq_len(take)]]
    i <- i + take
  }
  out
}

set.seed(SEED)
inp <- read.csv(IN_CSV, stringsAsFactors = FALSE)
inp$date <- as.Date(inp$date)
rows <- list()
for (asset in unique(inp$asset)) {
  if (!is.na(ONLY) && asset != ONLY) next
  d <- inp[inp$asset == asset & is.finite(inp$log_return), ]
  d <- d[order(d$date), ]
  n <- nrow(d)
  ts <- which(d$date >= OOS_START & seq_len(n) > WINDOW)
  ts <- ts[seq(1L, length(ts), by = STEP)]
  for (t in ts) {
    w <- d$log_return[(t - WINDOW):(t - 1)]
    pt <- tryCatch(FitML(spec = spec, data = w), error = function(e) NULL)
    p_point <- if (is.null(pt)) NA_real_ else p_end(pt, numeric(0))
    ps <- rep(NA_real_, B)
    for (b in seq_len(B)) {
      fb <- tryCatch(FitML(spec = spec, data = block_boot(w, BLOCK)), error = function(e) NULL)
      if (!is.null(fb)) ps[b] <- p_end(fb, w)
    }
    ok <- ps[is.finite(ps)]
    rows[[length(rows) + 1L]] <- data.frame(
      asset = asset, date = d$date[t - 1L], p_point = p_point,
      p_lo = if (length(ok)) unname(quantile(ok, 0.05)) else NA_real_,
      p_med = if (length(ok)) median(ok) else NA_real_,
      p_hi = if (length(ok)) unname(quantile(ok, 0.95)) else NA_real_,
      p_sd = if (length(ok) > 1) sd(ok) else NA_real_,
      n_ok = length(ok)
    )
    cat(sprintf("[boot] %s %s point=%.3f  band=[%.3f, %.3f] ok=%d/%d\n", asset,
                format(d$date[t - 1L]), p_point, rows[[length(rows)]]$p_lo,
                rows[[length(rows)]]$p_hi, length(ok), B), flush = TRUE)
  }
}
write.csv(do.call(rbind, rows), OUT_CSV, row.names = FALSE)
cat("[boot] wrote", OUT_CSV, "\n")
