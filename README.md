# cryptorisk

Comparative study of volatility / tail-risk models for crypto assets (VaR & ES),
evaluated out-of-sample with statistical significance.

This is **v2** of the BTC risk project. The v1 proof-of-concept lives in a
separate repo; its lessons are folded into [`docs/V2_PLAN.md`](docs/V2_PLAN.md),
which is the authoritative scope document.

## Status

**Phase 0 — setup.** Repo skeleton, pinned dependencies, config, CI, test
harness. Most modules are documented stubs that raise `NotImplementedError`
pointing at the relevant plan section. Two things are real:

- `cryptorisk.models.base` — the `PredictiveDist` / `Model` interface every
  model implements.
- `cryptorisk.backtest.coverage` — Kupiec, Christoffersen and the Basel traffic
  light (ported and hardened from v1), with known-answer tests.

See `docs/V2_PLAN.md` §8 for the phase plan.

## Layout

```
src/cryptorisk/
  data/       ingestion, realized measures, DuckDB store, quality checks
  models/     base interface + one module per model family
  backtest/   walk-forward engine + coverage / ES / scoring / PIT tests
  study/      orchestration, sub-periods, regime identification, report
  decision/   capital, limits, P&L attribution, hedge
config/study.yaml   assets, OOS period, windows, alphas, refit cadence, seeds
tests/              one known-answer test per statistical test
msgarch/            R scripts (MS-GARCH bridge, ported from v1)
```

`src/` layout: import as `from cryptorisk.backtest.coverage import kupiec_pof`.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
make test      # or: .venv/Scripts/python -m pytest
make lint      # ruff check
```

Requires Python ≥ 3.12 and R with the `MSGARCH` package for the MS-GARCH bridge
(not needed until Phase 2).

## Make targets

| target | does |
|---|---|
| `make data` | ingest daily + intraday + context + microstructure into the store |
| `make realized` | compute realized measures (RV, BV, jump) from 5-min bars |
| `make backtest` | run the walk-forward for all models and write `data/results/` |
| `make report` | full pipeline → `docs/methodology.md` tables and figures |
| `make test` / `make lint` | pytest / ruff |
