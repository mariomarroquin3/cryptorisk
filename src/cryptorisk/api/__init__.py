"""Read-only REST API over the study's pipeline outputs.

Independent of ``cryptorisk.dashboard`` (no Streamlit import) so it can run as
its own lightweight process -- ``make api`` / ``uvicorn cryptorisk.api.app:app``.
Same read-only contract as the dashboard: never writes to ``data/results/`` or
the store, except that ``data.today_forecast`` re-fits a model in-process for
a live one-step-ahead band (see its docstring).
"""
