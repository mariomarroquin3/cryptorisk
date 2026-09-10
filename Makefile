# cryptorisk — pipeline targets.
# On Windows use Git Bash / WSL, or run the underlying commands directly.

PY := .venv/Scripts/python
ifeq ($(OS),)
  PY := .venv/bin/python
endif

.PHONY: help install test lint fmt data realized msgarch backtest evaluate report clean

help:
	@echo "install   - create .venv and install (editable) with dev extras"
	@echo "test      - pytest"
	@echo "lint      - ruff check"
	@echo "fmt       - ruff format"
	@echo "data      - ingest daily/intraday/context/microstructure into the store (Phase 1)"
	@echo "realized  - compute realized measures from 5-min bars (Phase 1)"
	@echo "backtest  - walk-forward for all models -> data/results/ (Phase 3)"
	@echo "evaluate  - full evaluation battery -> data/results/eval_*.csv (Phase 3)"
	@echo "report    - full pipeline -> docs/methodology.md (Phase 6)"

install:
	python -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

fmt:
	$(PY) -m ruff format .

data:
	$(PY) -m cryptorisk.study.run_ingest

realized:
	$(PY) -m cryptorisk.data.realized

msgarch:
	$(PY) -m cryptorisk.study.run_msgarch

backtest:
	$(PY) -m cryptorisk.study.run_backtests

evaluate:
	$(PY) -m cryptorisk.study.run_evaluation

report:
	$(PY) -m cryptorisk.study.report

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
