"""Model implementations. One module per family (V2_PLAN §3).

Phase 2a: base interface + historical, ewma, garch (t / GJR / EGARCH), fhs,
garch_evt, jump.
Phase 2b: har (HAR-RV / HARQ), realized_garch, garch_x, caviar (SAV/AS + -X).
Phase 2c: msgarch_bridge - serves the R walk-forward's cached predictions.

``registry.all_models()`` returns all 16.
"""
