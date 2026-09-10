"""Model registry. ``study`` builds its model list from here.

Phase 2a: non-parametric, EWMA, single-regime GARCH, semiparametric tail, jump.
Phase 2b adds the realized-measure family, CAViaR(-X), GARCH-X.
Phase 2c adds MS-GARCH via the R bridge.
"""

from __future__ import annotations

from cryptorisk.models.base import Model
from cryptorisk.models.ewma import EWMA
from cryptorisk.models.fhs import FHS
from cryptorisk.models.garch import EgarchT, GarchT, GjrGarchT
from cryptorisk.models.garch_evt import GarchEVT
from cryptorisk.models.historical import HistoricalSimulation
from cryptorisk.models.jump import JumpDiffusion


def phase2a_models() -> list[Model]:
    return [
        HistoricalSimulation(),
        HistoricalSimulation(halflife=125, name="AWHS"),
        EWMA(),
        GarchT(),
        GjrGarchT(),
        EgarchT(),
        FHS(),
        GarchEVT(),
        JumpDiffusion(),
    ]


def all_models() -> list[Model]:
    return phase2a_models()
