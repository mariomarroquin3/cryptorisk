"""Load and validate ``config/study.yaml``.

The config is the single source of truth for every knob that affects results
(V2_PLAN §7). Access it via :func:`load_config`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _REPO_ROOT / "config" / "study.yaml"


@lru_cache(maxsize=4)
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Return the parsed study config as a plain dict.

    Cached by path. Pass an explicit ``path`` in tests.
    """
    p = Path(path) if path is not None else _DEFAULT_CONFIG
    if not p.exists():
        raise FileNotFoundError(f"study config not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    _require(cfg, ["seed", "assets", "sample", "walk_forward", "alphas", "evaluation"])
    _require(cfg["sample"], ["start", "oos_start"], where="sample")
    if not isinstance(cfg["assets"], list) or not cfg["assets"]:
        raise ValueError("config.assets must be a non-empty list")
    if not all(0.0 < a < 0.5 for a in cfg["alphas"]):
        raise ValueError("config.alphas must be tail probabilities in (0, 0.5)")
    return cfg


def repo_root() -> Path:
    return _REPO_ROOT


def _require(d: dict, keys: list[str], where: str = "config") -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValueError(f"{where} missing required keys: {missing}")
