"""Model registry. ``study`` builds its model list from here.

Phase 2a: non-parametric, EWMA, single-regime GARCH, semiparametric tail, jump.
Phase 2b adds the realized-measure family, CAViaR(-X), GARCH-X.
Phase 2c adds MS-GARCH via the R bridge.
"""

from __future__ import annotations

from cryptorisk.config import load_config, repo_root
from cryptorisk.models.base import Model
from cryptorisk.models.caviar import CAViaR
from cryptorisk.models.ewma import EWMA
from cryptorisk.models.fhs import FHS
from cryptorisk.models.garch import EgarchT, GarchT, GjrGarchT
from cryptorisk.models.garch_evt import GarchEVT
from cryptorisk.models.garch_x import GarchX
from cryptorisk.models.har import HAR
from cryptorisk.models.historical import HistoricalSimulation
from cryptorisk.models.jump import JumpDiffusion
from cryptorisk.models.msgarch_bridge import MSGarchBridge
from cryptorisk.models.realized_garch import RealizedGARCH


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


def phase2b_models(alphas: tuple[float, ...] | None = None) -> list[Model]:
    alphas = alphas or tuple(load_config()["alphas"])
    return [
        HAR(),
        HAR(harq=True),
        RealizedGARCH(),
        GarchX(),
        CAViaR(alphas, spec="SAV"),
        CAViaR(alphas, spec="AS"),
        CAViaR(alphas, spec="AS", exog=True),
    ]


def phase2c_models(alphas: tuple[float, ...] | None = None) -> list[Model]:
    cfg = load_config()
    alphas = alphas or tuple(cfg["alphas"])
    db = repo_root() / cfg["paths"]["store"]
    return [MSGarchBridge.from_store(db, alphas=alphas)]


def all_models() -> list[Model]:
    return phase2a_models() + phase2b_models() + phase2c_models()
