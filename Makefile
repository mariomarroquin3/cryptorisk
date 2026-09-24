# cryptorisk — pipeline targets.
# On Windows use Git Bash / WSL, or run the underlying commands directly.

PY := .venv/Scripts/python
ifeq ($(OS),)
  PY := .venv/bin/python
endif

.PHONY: help install test lint fmt data realized msgarch backtest evaluate subperiods regime-id regime-band decide report explain portfolio price-snapshot dashboard ops api web dev clean

help:
	@echo "install   - create .venv and install (editable) with dev extras"
	@echo "test      - pytest"
	@echo "lint      - ruff check"
	@echo "fmt       - ruff format"
	@echo "data      - ingest daily/intraday/context/microstructure + portfolio-basket dailies (Phase 1)"
	@echo "realized  - compute realized measures from 5-min bars (Phase 1)"
	@echo "backtest  - walk-forward for all models -> data/results/ (Phase 3)"
	@echo "evaluate  - full evaluation battery -> data/results/eval_*.csv (Phase 3)"
	@echo "explain   - RF-QR feature importance / ESS diagnostics -> explain_rf_*.csv"
	@echo "subperiods- sub-period re-eval + Giacomini-White CPA -> eval_subperiods.* (Phase 4)"
	@echo "regime-id - MS-GARCH regime identification from cached preds (Phase 4)"
	@echo "decide    - decision layer: capital / limits / PLA / hedge -> decision_*.csv (Phase 5)"
	@echo "report    - assemble docs/results.md + model_cards + figures (Phase 6)"
	@echo "portfolio - BTC+ETH basket VaR/ES with a copula tail -> portfolio_* (Phase 7)"
	@echo "price-snapshot - export data/results/price_history.parquet for the API (run after data/realized change; commit the diff before deploying)"
	@echo "dashboard - Streamlit live risk terminal over data/results/ + a live price feed"
	@echo "ops       - local Ops Center (jobs, data freshness, deploy checks) -> http://127.0.0.1:8502"
	@echo "api       - FastAPI read-only REST API over data/results/ -> http://localhost:8000/docs"
	@echo "web       - Next.js frontend (npm install first, in web/) -> http://localhost:3000"
	@echo "dev       - api + web together (Ctrl+C stops both) -> http://localhost:3000"

install:
	python -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev,dashboard,ml]"

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

fmt:
	$(PY) -m ruff format .

data:
	$(PY) -m cryptorisk.study.run_ingest

realized:
	$(PY) -m cryptorisk.study.run_realized

msgarch:
	$(PY) -m cryptorisk.study.run_msgarch

backtest:
	$(PY) -m cryptorisk.study.run_backtests

evaluate:
	$(PY) -m cryptorisk.study.run_evaluation

subperiods:
	$(PY) -m cryptorisk.study.subperiods

regime-id:
	$(PY) -m cryptorisk.study.regime_identification

regime-band:
	Rscript msgarch/bootstrap_pcrisis.R 25 90 data/results/msgarch_input.csv data/results/msgarch_pcrisis_band_BTC.csv BTC & \n	Rscript msgarch/bootstrap_pcrisis.R 25 90 data/results/msgarch_input.csv data/results/msgarch_pcrisis_band_ETH.csv ETH & wait

decide:
	$(PY) -m cryptorisk.study.run_decision

report:
	$(PY) -m cryptorisk.study.report

explain:
	$(PY) -m cryptorisk.study.run_explain

portfolio:
	$(PY) -m cryptorisk.study.run_portfolio

price-snapshot:
	$(PY) -m cryptorisk.study.export_price_history

dashboard:
	$(PY) -m streamlit run src/cryptorisk/dashboard/app.py

ops:
	$(PY) -m streamlit run src/cryptorisk/ops/app.py --server.address 127.0.0.1 --server.port 8502

api:
	$(PY) -m uvicorn cryptorisk.api.app:app --port 8000

web:
	npm --prefix web run dev

dev:
	@trap 'kill 0' INT TERM EXIT; \
	$(PY) -m uvicorn cryptorisk.api.app:app --port 8000 & \
	npm --prefix web run dev & \
	wait

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
