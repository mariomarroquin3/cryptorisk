"""Entry point: run the walk-forward for every (model, asset, window) and write
tidy results to ``data/results/`` (V2_PLAN §8, Phase 3).

    python -m cryptorisk.study.run_backtests
"""

from __future__ import annotations

from cryptorisk.config import load_config


def main() -> None:
    _cfg = load_config()
    raise NotImplementedError("study.run_backtests - Phase 3")


if __name__ == "__main__":
    main()
