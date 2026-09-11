# cryptorisk

A comparative study of volatility / tail-risk models for crypto assets — VaR and
Expected Shortfall for a BTC/ETH treasury position, evaluated **out-of-sample**
and ranked with **statistical significance**.

This is **v2** of the BTC risk project. v1 was a proof of concept (separate
repo); its lessons are folded into [`docs/V2_PLAN.md`](docs/V2_PLAN.md) — the
authoritative scope document — and the model docstrings. Operational detail for
contributors is in [`CLAUDE.md`](CLAUDE.md).

## Status

| Phase | | State |
|---|---|---|
| 0 | Setup — pinned deps, config, CI, test harness | ✅ |
| 1 | Data layer — DuckDB store, ingestion, realized measures, quality checks | ✅ |
| 2 | Models — 17 models + the walk-forward engine | ✅ |
| 3 | **Backtesting battery** — coverage + DQ + Basel, Acerbi–Székely ES, FZ0 loss + **Model Confidence Set**, PIT, vol-forecast eval | ✅ |
| 4 | Sub-periods (calm vs stress) + Giacomini–White CPA + MS-GARCH identification study | ✅ |
| 5 | Decision layer — FRTB ES-IMA capital, position limits, PLA test, perp hedge, estimation-risk band | ✅ |
| 6 | Results report (`docs/results.md`) + 17 model cards + figures | ✅ |
| 7 | Portfolio extension — 4-asset basket (BTC/ETH/SOL/BNB), copula tail dependence | ✅ |
| — | Dashboard — Streamlit risk terminal (live price + VaR/ES band, model comparison, capital, portfolio, regimes) | ✅ |
| — | API — FastAPI read-only REST API over the same outputs, independent of the dashboard | ✅ |

Data: BTC & ETH, daily 2018-01 → present, plus **1.8M 5-minute bars** for the
realized measures. Out-of-sample period is frozen at **2019-05-16 → present**
(≈ 2,674 days). The Phase-7 basket adds SOL & BNB daily returns (SOL reference
= Coinbase, since it is not on the CoinMetrics community tier); its joint OOS
starts later (SOL-bound, ~2020-08).

## The 17 models

| Family | Models |
|---|---|
| Non-parametric | Historical Simulation, Age-weighted HS |
| Exponential smoothing | EWMA (RiskMetrics, λ=0.94) |
| Single-regime GARCH | GARCH(1,1)-t, GJR-GARCH-t, EGARCH-t |
| Semiparametric tail | Filtered Historical Simulation, GARCH-EVT (McNeil–Frey) |
| Discontinuous | Merton jump-diffusion |
| Realized-measure | HAR-RV, HARQ, Realized GARCH |
| Exogenous / conditional | GARCH-X, CAViaR-SAV, CAViaR-AS, CAViaR-X-AS |
| Regime-switching | MS-GARCH (2-state, via an R bridge) |

Each returns a one-step predictive distribution through a common interface
(`Context` → `PredictiveDist`); a single model-agnostic engine runs the
walk-forward. See `CLAUDE.md` for the interface and each model's quirks.

## Quickstart

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"

make test          # ~150 known-answer + integration tests
make lint          # ruff

make data          # build the DuckDB store from source APIs   (~30 min)
make msgarch       # MS-GARCH walk-forward in R                 (~20 min, needs R + MSGARCH)
make backtest      # (model × asset × window) grid → data/results/backtests.parquet
make evaluate      # full battery → data/results/eval_*.csv + eval_summary.md
make subperiods    # stress/calm re-eval + Giacomini–White CPA → eval_subperiods.*
make regime-id     # MS-GARCH regime identification from cached preds
make decide        # decision layer → decision_*.csv + decision_summary.md
make report        # assemble docs/results.md + docs/model_cards/ + docs/figures/
make portfolio     # 4-asset basket VaR/ES with a copula tail → portfolio_* + docs/portfolio.md

.venv/Scripts/python -m pip install -e ".[dashboard]"
make dashboard     # Streamlit risk terminal → http://localhost:8501

.venv/Scripts/python -m pip install -e ".[api]"
make api           # FastAPI REST API → http://localhost:8000/docs
```

The headline result is the FZ0 ranking + 90% Model Confidence Set per
(asset, α); the full write-up is [`docs/results.md`](docs/results.md), the
mathematics is [`docs/methodology.tex`](docs/methodology.tex).

Requires **Python ≥ 3.12** (dev machine: 3.14) and, for `make msgarch` only, **R
with the `MSGARCH` package**. Dependencies are pinned exactly in
`pyproject.toml`.

## Layout

```
src/cryptorisk/
  config.py       config/study.yaml loader (single source of truth)
  data/           store.py (DuckDB) · ingest/ · realized.py · quality.py
  models/         base.py (interface) · registry.py · one module per family
  backtest/       engine.py (walk-forward) · coverage · es_tests · scoring (FZ0/DM/MCS/GW) · pit
  study/          run_ingest · run_msgarch · run_backtests · run_evaluation · vol_forecast_eval
                  · subperiods · regime_identification · run_decision
  decision/       capital (ES-IMA) · limits · pnl_attribution (PLA) · hedge · estimation_risk
  portfolio/      marginal (GARCH-t filter) · copula_var (k-dim Gaussian/t/Clayton)
  dashboard/      Streamlit terminal (optional extra: pip install -e ".[dashboard]")
  api/            FastAPI REST API, independent of dashboard/ (optional extra: pip install -e ".[api]")
config/study.yaml seed, assets, frozen OOS start, windows, alphas, MCS params, ...
msgarch/          R script + notes for the MS-GARCH bridge
tests/            one known-answer test per statistical test; test_integration is
                  store-gated
docs/             V2_PLAN.md (scope) · methodology.tex (maths) · results.md +
                  model_cards/ + figures/ (generated) · data_quality.md
```

`src/` layout: `from cryptorisk.backtest.coverage import kupiec_pof`.

## Dashboard

`make dashboard` (after `pip install -e ".[dashboard]"`) launches a Streamlit
terminal at `localhost:8501`:

- **Overview** — live BTC/ETH spot price (polled from Binance, no key) next to
  the FZ0-best model's VaR/ES band, re-fit on demand for a same-session
  forecast; a trailing price chart with the walk-forward VaR/ES cone and
  marked OOS violations.
- **Model Comparison** — FZ0 ranking + 90% MCS, coverage tests, Acerbi–Székely
  ES tests, Giacomini–White CPA, per (asset, α).
- **Portfolio** — the 4-asset basket's FZ0/MCS ranking (copula families vs.
  independence vs. direct univariate models) and live basket composition.
- **Capital & Decision** — the FRTB capital stack (point ES → model-risk
  add-on → estimation-risk add-on), position limits, perp hedge.
- **Regimes** — MS-GARCH crisis probability vs. realized vol, labelled
  in-sample vs. walk-forward per the caveat in `CLAUDE.md`.

Everything is read-only over `data/results/` and the store, except the live
"implied range" ticker, which re-fits the selected model in-session — clearly
badged **LIVE re-fit** vs. **FROZEN backtest** so it's never confused with the
study's own frozen, versioned numbers.

## API

`make api` (after `pip install -e ".[api]"`) launches a read-only FastAPI
server at `localhost:8000` (`/docs` for interactive Swagger). It mirrors the
dashboard's data as JSON — `/config`, `/models`, `/price/{asset}`,
`/forecast/{asset}` (live re-fit, same semantics as the dashboard's ticker),
`/models/comparison`, `/coverage`, `/es-tests`, `/gw-cpa`, `/backtests`,
`/portfolio/eval`, `/portfolio/composition`, `/capital`, `/estimation-risk`,
`/limits`, `/hedge`, `/regimes/{asset}` — for a custom frontend or any other
consumer. It's a separate process from the dashboard (no shared code, no
Streamlit import) so either can run without the other. No auth: a local,
read-only, personal-research tool, not meant to be exposed on an open network.

## Data & reproducibility

- Two independent daily price sources (Binance spot + CoinMetrics reference
  rate) are reconciled; the canonical return series uses the CoinMetrics USD
  reference. `prices_daily` rows carry a `vintage_ts` for point-in-time reads.
- Quality checks (`calendar_gap`, `extreme_return`, `source_divergence`,
  `few_intraday_bars`, …) run after every ingest; known events are explained in
  `config/quality_allowlist.yaml`, and the pipeline fails on any *unexplained*
  flag. See [`docs/data_quality.md`](docs/data_quality.md).
- Everything random takes a seed from `config/study.yaml`. `data/store/*.duckdb`
  and `data/results/` are gitignored — rebuild with the `make` targets.
