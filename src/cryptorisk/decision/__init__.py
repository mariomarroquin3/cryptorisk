"""Decision layer (V2_PLAN §6): translate the model comparison into
management numbers.

* :mod:`~cryptorisk.decision.capital`   -- FRTB ES-IMA capital + model-risk add-on
* :mod:`~cryptorisk.decision.limits`    -- position limit N* + a backtest of the framework
* :mod:`~cryptorisk.decision.pnl_attribution` -- FRTB PLA test (RTPL vs HPL)
* :mod:`~cryptorisk.decision.hedge`     -- perp hedge ratios, funding cost, ES reduction

Orchestrated by :mod:`cryptorisk.study.run_decision` (``make decide``).
"""
