# CLAUDE.md — cryptorisk (v2)

Operational guide. **`docs/V2_PLAN.md` is the scope document** (the *why*); this
is the *how*. Per-module detail lives in the module docstrings; per-phase
findings live in `data/results/*_summary.md` — don't duplicate either here.

## What this is

Comparative out-of-sample study of ~17 volatility / tail-risk models for crypto
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
  decision/            capital (FRTB ES-IMA), limits, pnl_attribution (PLA), hedge, estimation_risk
  dashboard/           Streamlit terminal over data/results/ + a live price feed (`make dashboard`)
  api/                 FastAPI read-only REST API, independent of dashboard/ (`make api`)
web/                   Next.js + TS + Tailwind frontend over api/, its own npm project (`make web`)
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

### The 17 models & their quirks

HS, AWHS · EWMA · GARCH-t, GJR-GARCH-t, EGARCH-t · FHS · GARCH-EVT ·
Jump-Diffusion · HAR-RV, HARQ · Realized-GARCH · GARCH-X · CAViaR-SAV,
CAViaR-AS, CAViaR-X-AS · MS-GARCH.

- GARCH family uses a **standardized** Student-t: rescale the scipy-t quantile
  by `sqrt((nu-2)/nu)` (`_dist.student_t_z`) — the factor v1 dropped.
- **EGARCH-t** has a vol *floor and ceiling* (`0.05·std` .. `20·std`): its
  log-variance forecast can collapse to ~0 → a *positive* VaR, or blow up.
  **HARQ** caps `h_next` at `30·median(RV)`. All parametric models fall back to
  the window's empirical quantile on any failure.
- **CAViaR** is fit **per alpha** → built with the study's alpha list; returns a
  `QuantileDist` (with a positive-VaR guard and quantile-crossing repair).
  **CAViaR-SAV** (symmetric news impact, no `(r)+`/`(r)-` split) added in the
  2026-09 hardening pass — `methodology.tex` had described it since Phase 6 but
  the registry only ever instantiated AS. Its rank swings hard by cell (BTC
  1%: 2/17; ETH 2.5%: 17/17, fails coverage+ES) — the asymmetry AS adds is not
  cosmetic for ETH.
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
make portfolio   # 4-asset basket VaR/ES with a copula tail -> portfolio_*
make price-snapshot  # data/results/price_history.parquet for the API (deploy-only; commit after data/realized changes)
make dashboard   # Streamlit terminal (pip install -e ".[dashboard]" first) -> localhost:8501
make api         # FastAPI REST API (pip install -e ".[api]" first) -> localhost:8000/docs
make web         # Next.js frontend (needs `make api` running; npm install first) -> localhost:3000
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
| 0-2 | setup · data layer · 17 models + engine + R bridge | ✅ |
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
  perp price in the store → perp return proxied by spot; with the proxy the
  ES-min hedge is degenerate (`h→1` zeroes the series) so `ratio_es_min` /
  `es_hedged` / `es_reduction` are set to NaN → rendered `n/a`. PLA
  `implied_rtpl` maps the realized outcome through the model's CDF, so Spearman
  ≈ 1 by construction — the KS is the discriminating metric.
- **Phase 5 estimation risk** `decision/estimation_risk.py` (`estimation_risk_table`
  in `run_decision` → `decision_estimation_risk.csv`, §8 of the report): bootstrap
  the **final** estimation window for 3 archetypes — HS (stationary block
  bootstrap), GARCH-t (draw θ*~N(θ̂,Σ̂) from `arch`'s `param_cov`, re-`forecast`
  with `params=θ*`, no refit), FHS (θ* for the vol path + residual resample).
  Prudent = 5th-pct ES draw; add-on = `es_capital(prudent) − es_capital(point)`
  at the Basel base multiplier. On this sample $32k–$73k (vs a ~$228k model-risk
  add-on) → second-order. Closes the "estimation risk not propagated" §9 caveat
  (now "bounded, not propagated").
- **Phase 6** `study.report` is a pure assembler: it reads `data/results/*.csv`
  + the store, writes `docs/results.md` (versioned), `docs/model_cards/*.md`
  (versioned, 16 + README) and `docs/figures/*.png` (**gitignored**). Per-model
  prose lives in `_MODEL_NOTES` in `report.py`; everything else is tabulated
  from the CSVs, so re-run `make report` after any pipeline re-run.
- **Phase 7** `study.run_portfolio` (`make portfolio`): fixed-weight basket,
  `config.portfolio.assets` (default BTC/ETH/SOL/BNB, equal weight) — a **separate
  list from the top-level single-asset `assets`**. Basket assets need only
  `returns_daily` in the store (copula marginals are plain GARCH-t, `Direct-*`
  are return-only) — no 5-min bars / realized / MS-GARCH. `make data` /
  `run_ingest` **auto-ingests the `portfolio.assets` not in `assets`
  daily-only** (unless `--assets` is narrowed or `--skip-portfolio-extras`), so
  the reproducibility chain works from a clean checkout. `portfolio/marginal.py`
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

- `data/store/*.duckdb`, `data/raw/`, `methodology.pdf` are **gitignored** —
  rebuild with the pipeline. **`data/results/` is versioned** (2026-09,
  deploy prep): the deployed API ships from this snapshot alone, since Render
  has no free persistent disk for the 100+MB store. Re-run the pipeline +
  `make price-snapshot` and commit the diff to refresh it.
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
- **`jump` column**: the tripower-quarticity constant `_MU_43` in `realized.py`
  was fixed earlier (was ~1.7× too big → jump Z over-fired) but the stored
  column stayed stale because `make realized` was a no-op (see below) — never
  actually recomputed until `study.run_realized` existed to run it. Refreshed:
  BTC/ETH jump-day share ~41.5% → ~32.6%/32.7%. Still nothing reads `jump`.
- **`make realized` was a silent no-op**: it shelled to
  `python -m cryptorisk.data.realized`, a pure-function module with no
  `__main__` (every orchestrator in this repo lives under `cryptorisk.study.*`
  — `data/*` modules never have their own CLI, matching `data/quality.py`,
  `data/ingest/*.py`). Added `study/run_realized.py` (reads cached `bars_5m`,
  recomputes `realized_daily`, no re-fetch) and pointed the Makefile target at
  it. `--assets` narrows it; safe to re-run (idempotent, same DELETE+INSERT
  pattern as every other store writer).
- FRED (fed funds / CPI) is blocked from some networks → those `context_daily`
  columns stay NULL; descriptive only, never VaR features.
- Quality: first ingest raises ~71 flags, **all allow-listed** (USDT/USD basis
  pre-2019-07, Binance outage days, COVID 2020-03-12). `run_ingest` exits
  non-zero on any *unexplained* flag.
- `test_integration` needs the store; skipped otherwise, so CI stays green
  without data.
- **`dashboard/`** (2026-09, `make dashboard`) is read-only over `data/results/`
  + the store, plus a polled Binance spot price (no key). The one exception is
  `dashboard/data.py::today_forecast`, which **re-fits** the selected model on
  the latest cached window for a live one-step-ahead band next to the live
  price -- explicitly badged "LIVE re-fit" vs. "FROZEN backtest" in the UI so
  it's never mistaken for the study's own frozen numbers. Falls back to the
  last stored backtest row if the live re-fit isn't supported (no ppf on a
  `QuantileDist` model just means no upper bound, not a failure -- shown as
  a one-sided "VaR floor" instead of a two-sided range). `streamlit`/`plotly`
  are an **optional** extra (`pip install -e ".[dashboard]"`), not in the core
  `dependencies` -- the study pipeline itself has no UI dependency.
- **`api/`** (2026-09, `make api`) is a FastAPI read-only REST API mirroring
  the dashboard's data (VaR/ES, FZ0/MCS, coverage/ES tests, portfolio,
  capital, regimes) plus a live price and `/forecast/{asset}` (same live
  re-fit semantics as the dashboard's ticker: `source: "live_refit"` vs.
  `"frozen_backtest"`). Deliberately does **not** import `cryptorisk.dashboard`
  or Streamlit -- `api/data.py` duplicates the small amount of read logic
  instead, so the two surfaces can run as fully independent processes. Its own
  optional extra (`pip install -e ".[api]"`: `fastapi`, `uvicorn`). No auth
  (read-only, so a public deploy risks availability, not data exposure);
  `CORS_ORIGINS` env var (comma-separated, defaults to the local dev ports)
  restricts which browser origins may call it. `/docs` for interactive
  Swagger. Caching is a hand-rolled in-process TTL dict (`api/cache.py`), not
  Streamlit's `st.cache_data` (unavailable outside a Streamlit run). Added a
  `/prices/{asset}` endpoint (date/close/log_return history) that neither the
  Streamlit dashboard nor the original API design needed -- the dashboard
  reads the store directly, but a decoupled frontend can't, so it's the one
  read the API had to grow to support `web/`.
  **Deploy prep (2026-09):** `api/data.py` used to query the DuckDB store
  directly for `load_price_window`/`regime_series` (fine locally; wrong for
  Render, which has no free persistent disk and can't be handed a
  100+MB gitignored file). `study/export_price_history.py`
  (`make price-snapshot`) now dumps just the daily columns those two
  endpoints need to `data/results/price_history.parquet`; `regime_series`
  reads the already-existing `data/results/msgarch_pred_{asset}.csv` instead
  of the store's `msgarch_predictions` table. The API now runs off
  `data/results/` alone, matching its own docstring, and has no DuckDB
  dependency at runtime. See `render.yaml` and the README's Deploy section.
  **Forward cone (2026-09):** `api/cone.py` re-fits Jump-Diffusion and
  GARCH-EVT live on the current window, extended to horizons `(1, 5, 10, 30)`
  days -- not a sqrt(H)-scaled single-day quantile, but each model's own
  correct multi-day math (Jump-Diffusion's Poisson-jump count and diffusion
  variance both scale exactly with H; GARCH-EVT's H-day variance is the sum
  of `arch`'s per-step forecasts, which mean-reverts, with the same fitted
  GPD tail rescaled to that variance). Independent of the `/forecast`
  `model` query param -- it's always these two models plus, if
  `data/results/msgarch_regime_params.csv` covers the asset, a third
  **labeled scenario** ("if the MS-GARCH crisis regime's own stationary vol
  applied and persisted"), *not* a forecast -- MS-GARCH's walk-forward regime
  probability has ~no OOS predictive power (see `msgarch_bridge`'s
  docstring), so it is never used as if it could predict which regime holds
  at day H. `msgarch_regime_params.csv` is written by
  `fit_msgarch_walkforward.R::regime_params()` from the *same* full-sample
  fit already computed for `prob_crisis_insample` (no extra R fitting cost) --
  re-run `make msgarch` to refresh it.
- **`web/`** (2026-09, `make web` from `cryptorisk/web/`) is a separate
  Next.js + TypeScript + Tailwind v4 app, a third, independent front-end over
  the same `api/` (not the dashboard). Client-rendered (no server-side data
  fetching) via SWR hooks in `web/lib/hooks.ts` polling `web/lib/api.ts`'s
  typed client against `NEXT_PUBLIC_API_BASE_URL` (`web/.env.local`, gitignored;
  `.env.example` is the committed template). Recharts for bar/area/line charts,
  `lightweight-charts` (TradingView's library) for the Overview page's
  historical price+VaR/ES band with breach markers. Its own
  `node_modules`/`package.json` -- not part of the Python package or its
  dependency groups; run `npm install` once in `web/`, then `npm run dev`
  (or `make web`).
  **Overview page panels (2026-09):** below the historical band, two small
  `ConeChart.tsx` panels (Jump-Diffusion, GARCH-EVT -- each its own chart, not
  overlaid, to stay readable) plot `/forecast`'s `cone` field: a continuous
  daily curve out to 30 days (a numeric x-axis, not the 4-point categorical
  axis it started as), converging on today's price. Below that,
  `RegimeDistribution.tsx` renders `regime_summary` as two Student-t density
  curves (mean 0, each regime's own fitted std/tail shape;
  `web/lib/studentT.ts` implements the density with a Lanczos gamma
  approximation) -- deliberately NOT another forecast line, since MS-GARCH's
  regime signal can't honestly support one (see `api/cone.py`'s docstring).
  **Visual design (2026-09, "Terminal Pro"):** `app/layout.tsx` loads
  JetBrains Mono and a display face (Space Grotesk, headings only) via
  `next/font/google` -- previously the mono font was only named in a
  `font-family` list and silently fell back to the OS default on any machine
  without it installed. `app/globals.css` adds a `.card` /
  `.card-interactive` (hover-lift, used by `MetricCard`) style reused by
  every chart/table container, plus a `pulse-dot` animation on the "Live
  re-fit" badge. The identical `Field()` label-wrapper duplicated across all
  5 pages was consolidated into `components/Field.tsx`.

## Review backlog (2026-09 full-project review — all six items fixed)

All six items are now closed (git log has each commit); none ever moved a
headline (FZ0/MCS) result — recomputes where needed confirmed byte-identical
or noise-level changes. Kept as a log of what to watch for after a similar
review, not open work:

- DM's HLN small-sample correction used the HAC lag instead of the horizon
  `h=1`; `jump.py`'s simulated drift double-counted an Itô term vs. its own
  MLE; `.to_numpy(bool)` upcast a non-finite-VaR `NaN` to `True`;
  `evaluate_gw_cpa` didn't drop degenerate-forecast days the way
  `evaluate_fz0_mcs` does.
- `vol_forecast_eval` used row-wise `dropna()` (a single model's NaN `sigma2`
  would shrink every model's sample); now column-wise like `evaluate_fz0_mcs`
  (drop that model). No-op on real data today — no model has a *partial* NaN
  `sigma2` (each is either 0/N or N/N) — but the row-wise version was a latent
  trap for one that someday does (e.g. a sparser MS-GARCH cache).
  `test_vol_forecast_drops_a_sparse_model_without_shrinking_the_others` locks
  it in.
- `CopulaVaR.fit_predict` paired each asset's `z_resid` by truncating to the
  shortest and slicing `[-L:]` — silently misaligning calendar days across
  assets if lengths ever differed (they don't today: every basket asset's
  window is a jointly-dropna'd, equal-length slice, and neither the `arch`
  GARCH(1,1) fit nor the EWMA fallback drops observations from an
  already-finite input). Now asserts the lengths match and degrades to the
  independence copula — rather than silently claiming a fitted dependence —
  on the one path that could violate it (`marginals=None`, i.e. not called
  through `run_portfolio`).
  `test_copula_var_falls_back_to_independence_on_mismatched_residual_lengths`
  reproduces the mismatch synthetically and locks in the fallback.

## Conventions

- Commit when work is done and tests are green. v2 commits use
  `git -c user.name="mariomarroquin3" -c user.email="mario.marroquin.2007@gmail.com"`.
- End commit messages with `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`.
- Keep the Makefile **LF** line endings (a Windows `write_text` once flipped it
  to CRLF, which breaks recipe tabs).
- Blunt honesty on what doesn't work — the "what the mathematics does not fix"
  section of methodology.tex is the model for that.
