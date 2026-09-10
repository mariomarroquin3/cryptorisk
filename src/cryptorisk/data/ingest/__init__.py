"""One module per source (V2_PLAN §2). Each writes to the store with a vintage.

* ``prices_daily``   - two independent daily price sources + reconciliation
* ``binance_klines`` - 5-minute bars from data.binance.vision
* ``context``        - macro + hashrate/difficulty (descriptive only)
* ``microstructure`` - perp funding, open interest, exchange netflows, stablecoin supply

Phase 1.
"""
