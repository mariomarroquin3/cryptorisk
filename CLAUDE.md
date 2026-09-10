# CLAUDE.md — cryptorisk (v2)

Operational guide. **`docs/V2_PLAN.md` is the scope document** (the *why*); this
is the *how*. Per-module detail lives in the module docstrings; per-phase
findings live in `data/results/*_summary.md` — don't duplicate either here.

## What this is

Comparative out-of-sample study of ~16 volatility / tail-risk models for crypto
(BTC + ETH), VaR & ES, ranked with statistical significance (Model Confidence
Set). v1 (`cubo-btc-risk`, separate repo) was the proof of concept.

## Environment

- **Python ≥ 3.12** (dev machine: 3.14). This repo has **its own venv**:
  `python -m venv .venv && .venv/Scripts/python -m pip install -e ".[dev]"`.
  Do **not** use v1's venv. Deps are **pinned exactly** in `pyproject.toml`.
- **R + `MSGARCH` 2.51** — only for the MS-GARCH bridge. `Rscript` on PATH.
- `src/` layout: `from cryptorisk.backtest.coverage import kupiec_pof`.
- CI runs `ruff check` + `pytest` only (not `ruff format`). MiKTeX `pdflatex`
  for `docs/methodology.tex`.

## Layout

```
src/cryptorisk/
  config.py            load + validate config/study.yaml (single source of truth)
  data/
    store.py           DuckDB schema + idempotent writers; SCHEMA_VERSION in meta
    ingest/            prices_daily, binance_klines (5m), context, microstructure
    realized.py        RV / BV / semivariance / jump (BNS) / realized quarticity
    quality.py         flag checks + allow-list (config/quality_allowlist.yaml)
  models/
    base.py            Context, PredictiveDist (Parametric/Empirical/Quantile), Model
    _dist.py _gpd.py _util.py    shared helpers
    <family>.py        historical ewma garch fhs garch_evt jump har realized_garch
                       garch_x caviar msgarch_bridge
    registry.py        all_models() = phase2a + 2b + 2c
  backtest/
    engine.py          walk_forward(): the model-agnostic OOS loop
    coverage.py        Kupiec, Christoffersen, Engle-Manganelli DQ, Basel
    es_tests.py        Acerbi-Szekely Z1/Z2 (+ asymptotic p-value)
    scoring.py         FZ0, QLIKE, Diebold-Mariano, MCS, Giacomini-White CPA
    pit.py             PIT cleaning + Berkowitz LR
  study/               orchestrators, one per `make` target (see Pipeline)
  decision/            capital (FRTB ES-IMA), limits, pnl_attribution (PLA), hedge
config/study.yaml      seed, assets, frozen OOS start, windows, alphas, MCS params,
                       sub-periods, decision-layer knobs
msgarch/               R scripts + notes for the bridge
tests/                 known-answer per statistical test; test_integration store-gated
```

## Model interface (models/base.py)

`Model`: `name: str` + `fit_predict(ctx: Context) -> PredictiveDist`. Engine
calls it once per OOS day, queries the dist at each `alpha`.

- **Conventions**: log-returns; `alpha` = lower-tail prob (0.025 → 97.5% VaR);
  `var`/`es` are return levels, normally **negative**; violation on day t is
  `realized_t < var_t`. **No look-ahead**: the model sees returns/realized for
  `[window_start, t-1]`, forecasts day t, is scored on `r[t]`.
- `Context`: `returns` (no NaN), `dates`, `asof` (last obs), `asset`, optional
  `realized` dict (`rv bv rsv_pos rsv_neg jump rq`) and `exog` dict.
- Dists: `ParametricDist` (loc/scale + standardized ref), `EmpiricalDist`
  (weighted sample), `QuantileDist` ((VaR,ES) pairs only — CAViaR, MS-GARCH
  bridge; no `cdf`/`sigma2`). `sigma2/cdf/ppf` may raise `NotImplementedError`.

### The 16 models & their quirks

HS, AWHS · EWMA · GARCH-t, GJR-GARCH-t, EGARCH-t · FHS · GARCH-EVT ·
Jump-Diffusion · HAR-RV, HARQ · Realized-GARCH · GARCH-X · CAViaR-AS,
CAViaR-X-AS · MS-GARCH.

- GARCH family uses a **standardized** Student-t: rescale the scipy-t quantile
  by `sqrt((nu-2)/nu)` (`_dist.student_t_z`) — the factor v1 dropped.
- **EGARCH-t** has a vol *floor and ceiling* (`0.05·std` .. `20·std`): its
  log-variance forecast can collapse to ~0 → a *positive* VaR, or blow up.
  **HARQ** caps `h_next` at `30·median(RV)`. All parametric models fall back to
  the window's empirical quantile on any failure.
- **CAViaR** is fit **per alpha** → built with the study's alpha list; returns a
  `QuantileDist` (with a positive-VaR guard and quantile-crossing repair).
- AR(1) recursions in CAViaR / Realized-GARCH / GARCH-X use `scipy.signal.lfilter`,
  not Python loops (Realized-GARCH: 751s → 37s/window).
- **MS-GARCH** runs its whole walk-forward in **R** (`make msgarch`), caches
  per-day predictions in `store.msgarch_predictions`; `MSGarchBridge.fit_predict`
  looks up `(asset, prev_date)` and falls back to empirical on a miss. Its
  `refit_every=20` lives in the R script. The walk-forward regime probability
  has ~no OOS signal; only the full-sample `prob_crisis_insample` tracks
  volatility, and it's for the regime study only (V2_PLAN §5.5).

## Pipeline

```bash
make data        # ingest -> store + data/results/quality_report*.csv
make msgarch     # R MS-GARCH walk-forward -> store.msgarch_predictions   (~20 min, needs R)
make backtest    # walk-forward grid -> data/results/backtests.parquet (+ _summary.csv)
make evaluate    # full battery -> eval_{coverage,es,fz0_mcs,density,volforecast}.csv + eval_summary.md
make subperiods  # stress/calm re-eval + Giacomini-White CPA -> eval_subperiods.* + eval_gw_cpa.csv
make regime-id   # MS-GARCH regime-prob vs vol state -> regime_identification.*
make decide      # FRTB capital / limits / PLA / hedge -> decision_*.csv + decision_summary.md
make report      # assemble docs/results.md + docs/model_cards/ + docs/figures/
make test lint fmt
```

- **Frozen OOS start** (`config sample.oos_start` = 2019-05-16): never tune it
  against results. OOS ≈ 2,674 days/asset.
- `run_backtests --models <subset>` **merges** into the existing parquet.
- `backtests.parquet` row = (date, asset, model, window, alpha): `date` is the
  forecast target; `var/es/sigma2/pit` are that day's forecast; `realized` is
  `r[date]`; `violation = realized < var`.
- **After editing a model**: re-run `backtest --models <that model>` then
  `evaluate` (and `decide`) — the parquet/CSVs are stale otherwise.

## Phase status (V2_PLAN §8)

| # | scope | state |
|---|---|---|
| 0-2 | setup · data layer · 16 models + engine + R bridge | ✅ |
| 3 | evaluation battery (`run_evaluation`, `vol_forecast_eval`) | ✅ |
| 4 | sub-periods + Giacomini-White CPA + MS-GARCH regime identification | ✅ |
| 5 | decision layer (`run_decision`) | ✅ |
| 6 | results report + model cards + figures (`study/report.py`, `make report`) | ✅ |
| 7 | portfolio extension — 4-asset basket (BTC/ETH/SOL/BNB), copula tail (`run_portfolio`, `make portfolio`) | ✅ |

**All phases done**, including the optional Phase 7 (now a 4-asset basket).
Further work is hardening (the notes below).

Non-obvious, still-relevant facts:

- **Phase 3** `eval_fz0_mcs` drops days where *any* model emits a degenerate
  forecast (ES ≥ −1e-6 or VaR ≥ 0), count in `n_degenerate`. Acerbi-Szekely
  p-values are the **asymptotic-normal** approximation; the simulation version
  needs models to expose per-day predictive draws (not built).
- **Phase 4** regime sweep (`fit_msgarch_regime_windows.R`, `regime-id --run-r`)
  is a multi-hour R job — **not run**; `run()` uses cached results if present.
- **Phase 5** `run_decision`: capital *amount* uses the 97.5% ES, but `m_c`
  (Basel traffic-light) is a **99% concept** → exception count comes from the
  α=0.01 violations. `hedge.funding_carry_*` sign: funding > 0 ⇒ longs pay
  shorts, and the hedge is *short* the perp, so positive carry = **income**. No
  perp price in the store → perp return proxied by spot (ratios ≈ 1, basis risk
  unavailable). PLA `implied_rtpl` maps the realized outcome through the model's
  CDF, so Spearman ≈ 1 by construction — the KS is the discriminating metric.
- **Phase 6** `study.report` is a pure assembler: it reads `data/results/*.csv`
  + the store, writes `docs/results.md` (versioned), `docs/model_cards/*.md`
  (versioned, 16 + README) and `docs/figures/*.png` (**gitignored**). Per-model
  prose lives in `_MODEL_NOTES` in `report.py`; everything else is tabulated
  from the CSVs, so re-run `make report` after any pipeline re-run.
- **Phase 7** `study.run_portfolio` (`make portfolio`): fixed-weight basket,
  `config.portfolio.assets` (default BTC/ETH/SOL/BNB, equal weight) — a **separate
  list from the top-level single-asset `assets`**. Basket assets need only
  `returns_daily` in the store (copula marginals are plain GARCH-t, `Direct-*`
  are return-only) — no 5-min bars / realized / MS-GARCH. `portfolio/marginal.py`
  = GARCH(1,1)-t vol filter (refit every `config.portfolio.refit_every` days);
  `portfolio/copula_var.py` = FHS residual inversion + a **k-dimensional** copula
  (independence / gaussian / student_t [fixed df] / clayton) via
  `statsmodels.distributions.copula`, MC-aggregated. gaussian / student_t use a
  full PSD-repaired correlation matrix (`_corr_from_param` + `_nearest_psd`);
  `fit_corr_param` returns a scalar for k=2 and a matrix for k≥3. Also runs a few
  univariate models straight on the basket series (`Direct-<model>`) via the
  normal engine. Evaluated with the Phase-3 battery (`asset="PORTFOLIO"`).
  Clayton falls back to independence when the fitted theta ≤ 0. The `## Read`
  section of `docs/portfolio.md` is **generated from the eval numbers**
  (`_read_bullets`), not hardcoded. Writes `portfolio_backtests.parquet`,
  `portfolio_eval.csv`, `portfolio_subperiods.csv`, `docs/portfolio.md`.
- **Phase 7 data**: SOL is not on the CoinMetrics community tier (HTTP 403), so
  its reference rate is **Coinbase `SOL-USD`** daily candles
  (`prices_daily._REFERENCE` / `fetch_reference_daily`; SOL history starts
  2020-08 on Binance spot, 2021-06 on Coinbase). BNB uses CoinMetrics. The
  basket's joint OOS therefore starts later than BTC+ETH's (SOL-bound).
  `run_ingest` stores the reference under its true provider label; quality
  re-reads pick `source <> 'binance'`.

## Gotchas

- `data/store/*.duckdb`, `data/results/`, `data/raw/`, `*.parquet`,
  `methodology.pdf` are **gitignored** — rebuild with the pipeline. Source,
  config, docs, tests are versioned.
- **Store schema change**: bump `store.SCHEMA_VERSION` and add
  `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` to `_DDL` (DuckDB won't add columns
  via `CREATE TABLE IF NOT EXISTS`). `asof` is a **DuckDB reserved keyword** —
  the msgarch column is `prev_date`.
- A realized column reaches models only if it's in **both** `engine._REALIZED_COLS`
  **and** the `run_backtests` SELECT. Models currently consume only `rv` and `rq`
  (HAR/HARQ) / `rv` (Realized-GARCH, GARCH-X, CAViaR-X). `bv` and `jump` are
  computed and stored but **nothing downstream reads them**.
- `config/study.yaml` `refit_every` is **documentation only** — `run_backtests`
  refits every model every day; MS-GARCH's cadence is in the R script.
- **`jump` column**: fixed the tripower-quarticity constant `_MU_43` in
  `realized.py` (was ~1.7× too big → jump Z over-fired). Nothing reads `jump`,
  so results are unaffected, but the **stored column is stale until
  `make realized` re-runs**.
- FRED (fed funds / CPI) is blocked from some networks → those `context_daily`
  columns stay NULL; descriptive only, never VaR features.
- Quality: first ingest raises ~71 flags, **all allow-listed** (USDT/USD basis
  pre-2019-07, Binance outage days, COVID 2020-03-12). `run_ingest` exits
  non-zero on any *unexplained* flag.
- `test_integration` needs the store; skipped otherwise, so CI stays green
  without data.
- `methodology.tex` describes CAViaR-**SAV**, but only the **AS** spec is in
  `registry.py` (→ reconcile the doc or add SAV before Phase 6).

## Review backlog (known, not yet fixed)

From the 2026-09 full-project review; none affect a headline (FZ0 / MCS)
result, all are latent or touch a secondary column.

- **`scoring.diebold_mariano` HLN correction** uses the HAC truncation `lag`
  where the Harvey–Leybourne–Newbold formula wants the forecast horizon `h`
  (= 1 here). Shrinks the DM stat (over-conservative). Only `dm_vs_best_p` uses
  it; the MCS does not.
- **`jump.py` drift inconsistency**: the MLE centres the n=0 mixture component
  at `mu_base − λk̄`; `fit_predict` simulates with an extra `−0.5σ²`
  (`mu_base − 0.5σ² − λk̄`). ~8 bp/day for BTC; nudges Jump-Diffusion VaR/ES
  slightly more negative than the fitted model.
- **`.to_numpy(bool)` on `violation`** in `evaluate_coverage` /
  `run_decision.{capital,limits}_table`: a non-finite VaR → `violation` NaN →
  `NaN.astype(bool) == True` (counts a non-forecast as a breach). Not triggered
  today (every model falls back to a finite forecast; 0 nulls in the parquet).
  Prefer `(realized < var) & isfinite(var)`.
- **`evaluate_gw_cpa`** (subperiods) does not drop degenerate-forecast days
  (`v<0 & e<-1e-6`) the way `evaluate_fz0_mcs` does. 3 positive-VaR rows exist
  (EGARCH-t); harmless only because EGARCH-t is never a per-cell FZ0 best nor a
  `_GW_CHALLENGERS` member.
- **`vol_forecast_eval`** uses row-wise `dropna()` (drop the day for every
  model if one has a NaN `sigma2`); `evaluate_fz0_mcs` uses column-wise
  `dropna(axis=1)` (drop the model). A sparse `sigma2` (e.g. MS-GARCH bridge)
  silently shrinks the QLIKE-MCS sample for everyone.
- **`copula_var` residual alignment**: `marginal._ewma_fallback` compresses out
  non-finite residuals, and the copula pairs residuals by recency (`x[-L:]`),
  so an interior drop mis-aligns the pseudo-obs from that point back. Low
  probability (`arch` `std_resid` rarely has interior NaNs).

## Conventions

- Commit when work is done and tests are green. v2 commits use
  `git -c user.name="mariomarroquin3" -c user.email="mario.marroquin.2007@gmail.com"`.
- End commit messages with `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`.
- Keep the Makefile **LF** line endings (a Windows `write_text` once flipped it
  to CRLF, which breaks recipe tabs).
- Blunt honesty on what doesn't work — the "what the mathematics does not fix"
  section of methodology.tex is the model for that.
