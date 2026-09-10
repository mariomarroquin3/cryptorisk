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
    coverage.py        Kupiec, Christoffersen, Basel traffic light        (done)
    es_tests.py scoring.py pit.py                                        (Phase 3 stubs)
  study/
    run_ingest.py      `make data`  — build the store, run quality checks
    run_msgarch.py     `make msgarch` — export returns, run R, load predictions
    run_backtests.py   `make backtest` — (model x asset x window) grid -> parquet
    ...                report, subperiods, regime_identification, vol_forecast_eval  (later)
  decision/            capital, limits, pnl_attribution, hedge            (Phase 5 stubs)
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
- Next: **Phase 3** — coverage battery (+ Engle-Manganelli DQ), Acerbi-Székely
  ES tests, FZ0 loss + Diebold-Mariano + **Model Confidence Set**, PIT/Berkowitz,
  vol-forecast eval (QLIKE vs RV).
- Then 4 (sub-periods + regime identification), 5 (decision layer), 6 (report).
