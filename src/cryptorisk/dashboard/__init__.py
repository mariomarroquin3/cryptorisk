"""Streamlit front-end over the study's pipeline outputs.

Read-only: this package never fits a model to change a stored result, and
never writes to ``data/results/`` or the DuckDB store. The one exception is
:func:`data.today_forecast`, which re-fits the current FZ0-best model on the
latest cached window to show a live, one-step-ahead price band next to a
polled spot price -- clearly labelled as such, never mixed into the study's
own (frozen, versioned) backtest numbers.

Run with ``make dashboard`` (see CLAUDE.md).
"""
