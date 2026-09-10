"""One module per source (V2_PLAN §2). Each returns tidy DataFrames; the
orchestrator (``cryptorisk.study.run_ingest``) writes them to the store.

* ``prices_daily``   - Binance spot + CoinMetrics reference, reconciled
* ``binance_klines`` - 5-minute bars from data.binance.vision (streamed monthly)
* ``context``        - hashrate/difficulty + SPX/DXY/fed/CPI (descriptive only)
* ``microstructure`` - perp funding, open interest (partial history)
"""
