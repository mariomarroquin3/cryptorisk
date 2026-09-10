# CLAUDE.md — cryptorisk (v2)

Operational guide for this repo. **`docs/V2_PLAN.md` is the scope document** —
read it for the *why*. This file is the *how*.

## What this is

A comparative study of volatility / tail-risk models for crypto assets (VaR &
ES), evaluated out-of-sample with statistical significance. v1 (a separate repo,
`cubo-btc-risk`) was the proof of concept; its lessons are folded into
`docs/V2_PLAN.md` and the model docstrings.

## Environment

- **Python ≥ 3.12** (dev machine runs 3.14). This repo has **its own venv**:
  `python -m venv .venv && .venv/Scripts/python -m pip install -e ".[dev]"`.
  Do **not** use v1's venv.
- Dependencies are **pinned exactly** in `pyproject.toml`. Bump in a dedicated
  commit with CI green.
- **R** with the `MSGARCH` 2.51 package — only for the MS-GARCH bridge
  (Phase 2c). Not needed for anything else. `Rscript` must be on PATH.
- `src/` layout: import as `from cryptorisk.backtest.coverage import kupiec_pof`.

## Layout

```
src/cryptorisk/
  config.py            load + validate config/study.yaml (single source of truth)
  data/
    store.py           DuckDB schema + writers (idempotent). schema_version in meta.
    _http.py           retrying session for the ingestors
    ingest/            prices_daily, binance_klines (5m), context, microstructure
    realized.py        RV / BV / semivariance / jump (BNS) / realized quarticity
    quality.py         flag checks + allow-list (config/quality_allowlist.yaml)
  models/
    base.py            Context, PredictiveDist (+ ParametricDist / EmpiricalDist /
                       QuantileDist), Model(Protocol)
    _dist.py _gpd.py _util.py   shared helpers
    <one module per family>     historical, ewma, garch, fhs, garch_evt, jump,
                                har, realized_garch, garch_x, caviar, msgarch_bridge
    registry.py        phase2a_models() / phase2b_models() / phase2c_models() / all_models()
  backtest/
    engine.py          walk_forward(): the model-agnostic OOS loop
    coverage.py        Kupiec, Christoffersen, Engle-Manganelli DQ, Basel   (done)
    es_tests.py        Acerbi-Szekely Z1/Z2 (+ asymptotic p-value)          (done)
    scoring.py         FZ0, QLIKE, Diebold-Mariano, MCS, Giacomini-White CPA (done)
    pit.py             PIT cleaning + Berkowitz LR                          (done)
  study/
    run_ingest.py      `make data`  — build the store, run quality checks
    run_msgarch.py     `make msgarch` — export returns, run R, load predictions
    run_backtests.py   `make backtest` — (model x asset x window) grid -> parquet
    run_evaluation.py  `make evaluate` — full battery -> data/results/eval_*.csv + eval_summary.md
    vol_forecast_eval.py  QLIKE vs RV, QLIKE-MCS, Mincer-Zarnowitz (called by run_evaluation)
    subperiods.py      `make subperiods` — stress/calm re-eval + Giacomini-White CPA
    regime_identification.py  `make regime-id` — MS-GARCH regime-prob vs vol state
    report.py          `make report` — tables/figures for the write-up          (Phase 6)
    run_decision.py   `make decide` — capital / limits / PLA / hedge -> decision_*.csv
  decision/           capital (ES-IMA), limits, pnl_attribution (PLA), hedge   (done)
config/study.yaml      seed, assets, frozen OOS start, windows, alphas, refit cadence,
                       sub-periods, MCS params, decision-layer knobs
msgarch/               R script + notes for the bridge
tests/                 one known-answer test per statistical test; test_integration
                       is store-gated (skips if data/store/*.duckdb absent)
```

## The model interface (models/base.py)

Every model implements `Model`: `name: str` + `fit_predict(ctx: Context) ->
PredictiveDist`. The engine calls it once per OOS day, then queries the returned
dist at each configured `alpha`.

- **Conventions**: log-returns; `alpha` is a lower-tail probability (0.025 ->
  the 97.5% VaR); `var(alpha)` / `es(alpha)` are return levels, normally
  negative; a violation on day t is `realized_t < var_t`.
- `Context`: `returns` (window, no NaN), `dates`, `asof` (last obs), `asset`,
  optional `realized` dict (`rv, bv, rsv_pos, rsv_neg, jump, rq`) and `exog` dict.
- Three concrete dists: `ParametricDist` (location/scale + standardized ref
  dist), `EmpiricalDist` (weighted sample; HS/FHS/jump), `QuantileDist`
  (specific (VaR,ES) pairs only; CAViaR, MS-GARCH bridge — no `cdf`/`sigma2`).
- `sigma2()`, `cdf()`, `ppf()` may raise `NotImplementedError`; that is valid.

## The 16 models (registry.all_models())

A: HS, AWHS · EWMA · GARCH-t, GJR-GARCH-t, EGARCH-t · FHS · GARCH-EVT ·
Jump-Diffusion · HAR-RV, HARQ · Realized-GARCH · GARCH-X · CAViaR-AS,
CAViaR-X-AS · MS-GARCH.

- GARCH family uses a **standardized** Student-t (rescale the scipy-t quantile by
  `sqrt((nu-2)/nu)` — the bug v1 got wrong). Empirical fallback + counter.
- EGARCH and HARQ have a **vol ceiling** (fall back / clamp when `sigma_next` or
  `h_next` blows past ~20-30x the sample vol / median RV).
- CAViaR is fit **per alpha**, so it is built with the study's alpha list
  (`registry` passes it); it returns a `QuantileDist`. AR(1) recursions in
  CAViaR / Realized-GARCH / GARCH-X are solved with `scipy.signal.lfilter`, not
  Python loops (Realized-GARCH went 751s -> 37s per window).
- **MS-GARCH** does its whole walk-forward in **R** (`run_msgarch`), caches
  per-day predictions in `store.msgarch_predictions`, and `MSGarchBridge` just
  serves them (`fit_predict` looks up `(asset, prev_date)`). Unknown dates / no
  store / no R run -> empirical fallback. The walk-forward regime probability is
  a weak OOS signal; only the full-sample `prob_crisis_insample` tracks
  volatility, and it's for the regime study, not the VaR backtest
  (V2_PLAN §5.5).

## Pipeline

```bash
make data       # ingest -> store; writes data/results/quality_report*.csv
make msgarch    # R MS-GARCH walk-forward -> store.msgarch_predictions  (~20 min, needs R)
make backtest   # walk-forward grid -> data/results/backtests.parquet (+ summary)
make test lint  # pytest / ruff
```

`run_backtests` with a `--models` subset **merges** into the existing parquet
(keeps other models' rows). The OOS start (`config sample.oos_start`, currently
2019-05-16) is **frozen** — don't tune it against results.

## Gotchas

- `data/store/*.duckdb`, `data/results/`, `data/raw/` are **gitignored** —
  rebuild with the pipeline. Source, config, docs and tests are versioned.
- Store schema changes: bump `store.SCHEMA_VERSION` and add
  `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` to `_DDL` (DuckDB won't add columns
  via `CREATE TABLE IF NOT EXISTS`). `asof` is a **DuckDB reserved keyword** —
  the msgarch column is `prev_date`.
- A realized column only reaches models if it's in **both**
  `engine._REALIZED_COLS` **and** the `run_backtests` SELECT.
- Data currency: `dataset` ends whenever `make data` last ran; CoinMetrics is a
  daily reference rate (current). FRED (fed funds / CPI) is blocked from some
  networks -> those context columns stay NULL; they're descriptive only.
- Quality: 71 flags on the first ingest, **all allow-listed** (USDT/USD basis
  pre-2019-07, Binance outage days, COVID 2020-03-12). `run_ingest` exits
  non-zero on any *unexplained* flag.
- `test_integration` needs the store; it's skipped otherwise, so CI (no store)
  stays green on `make test` alone.

## Phase status (V2_PLAN §8)

- 0 setup · 1 data · 2a engine + 9 models · 2b realized family + CAViaR + GARCH-X
  · 2c MS-GARCH bridge — **done**.
- **Phase 3 — done**: `run_evaluation` runs the full battery over
  `backtests.parquet` and writes `eval_coverage.csv`, `eval_es.csv`,
  `eval_fz0_mcs.csv` (the headline), `eval_density.csv`, `eval_volforecast.csv`,
  `eval_summary.md`. `make evaluate`.
  - `eval_fz0_mcs` drops days where any model emits a degenerate forecast
    (ES ≥ -1e-6 or VaR ≥ 0) so one bad row can't dominate a mean FZ0; the count
    is in `n_degenerate`.
  - Acerbi-Székely p-values are the **asymptotic normal** approximation
    (`z2_pvalue_asymptotic`); the simulation version needs models to expose
    per-day predictive draws — a later addition.
  - Headline as of first run: **Realized-GARCH** wins FZ0 in 3 of 4 (asset, α)
    cells, HAR-RV wins ETH α=0.025; the 90% MCS is wide (13-15 of 16) — low
    power to separate the middle of the pack, EGARCH-t is the only clear
    exclusion.
- **EGARCH-t fix (garch.py)**: the vol guard now has a *floor*
  (`0.05 * r.std()`), not only a ceiling — a collapsed log-variance forecast was
  producing σ≈0 hence a *positive* VaR on ~60/2674 days. Re-run
  `run_backtests --models EGARCH-t` then `run_evaluation` after touching it.
- **Phase 4 — done** (the parts that need no new R):
  - `study.subperiods` (`make subperiods`): re-runs FZ0+MCS / coverage / ES
    inside each `config.evaluation.subperiods` window + full OOS, and a
    Giacomini-White CPA test (`scoring.giacomini_white`) of `L_best - L_chal`
    with conditioner `[1, z(log RV_{t-1})]` plus a HAC t-test on the RV-state
    slope. Writes `eval_subperiods.{csv,md}`, `eval_gw_cpa.csv`.
    Finding: the sub-period MCS is all-in everywhere (40-92 trading days → no
    power); the ranking does not detectably change across regimes.
  - `study.regime_identification` (`make regime-id`): from the cached
    `msgarch_pred_<asset>.csv`, correlations of each regime-probability series
    with |r| / RV / RV21. **corr(|r|): BTC walk-forward 0.01 vs full-sample
    0.70 (Spearman 0.91); ETH -0.03 vs 0.55.** Quantifies methodology.tex §9(i).
  - `msgarch/fit_msgarch_regime_windows.R` + `regime_identification --run-r`:
    the W ∈ {500..2000, expanding} × {free, arch} sweep. **Not run** — multi-hour
    R job; `run()` uses cached results if `regime_identification_windows.csv`
    exists, else just the W=500 analysis.
- **Phase 5 — done**: `study.run_decision` (`make decide`) turns the MCS models
  into management numbers, writing `decision_{capital,limits,pla,hedge}.csv` +
  `decision_summary.md`.
  - `decision/capital.py`: `capital = m_c*N*|ES_97.5,1d|*sqrt(LH)`. The capital
    *amount* uses the 97.5% ES; `m_c` (Basel base + traffic-light add-on) is a
    **99% backtesting** concept, so `run_decision.capital_table` counts
    exceptions from the α=0.01 violation series, not the 97.5% slice. Post-fix
    most models are m_c=1.5 (green at 99%); EWMA/HARQ/CAViaR-AS/GARCH-EVT are
    amber (1.9). `es_horizon_bootstrap` (stationary block bootstrap of LH-day
    compounded returns) is the √time cross-check; `model_risk_addon` = capital
    spread across the MCS.
  - `decision/limits.py`: `N* = budget/|ES_99,1d|`; `backtest_framework` reports
    bind rate, budget-breach rate, `flagged_breach_rate` (breach where that
    day's ES already exceeded budget -- same-day, since ES_t is the t-1 forecast
    for t), ES-exceedance rate (below α: ES < VaR), worst loss.
  - `decision/pnl_attribution.py`: `pla_test` (Spearman + KS, FRTB-style zones).
    RTPL = realized outcome mapped through the model's predictive CDF
    (`implied_rtpl`), so Spearman is ~1 by construction and the KS is the
    discriminating metric -- a known limitation of adapting PLA to a single-
    risk-factor VaR study. `attribute_pnl` keeps `jump_var_share=0` (real
    diffusion/jump split needs stored jump params, not in the parquet).
  - `decision/hedge.py`: MV + ES-minimising hedge ratios, `funding_carry_annualised`
    (= mean 8h funding * 3 * 365). **Sign**: funding > 0 => longs pay shorts, and
    the hedge is *short* the perp, so a positive carry is *income*. **No perp
    price in the store** -> perp return proxied by spot; ratios sit at 1.0,
    basis risk reported unavailable. Real output: the hedge *earns*
    ~+$116k/yr (BTC) / +$138k/yr (ETH) in carry while removing the directional ES.
  - Headline: model-risk add-on ~$228k (BTC) / ~$185k (ETH) on a $1M notional --
    picking within the "statistically tied" MCS moves required capital ~50%.
    Thin-tailed models (EWMA/HARQ/HAR-RV) give a bigger N* that then breaches
    the budget ~1.4% of days vs ~0.3% for the GARCH family.
- Then 6 (report).
