"""Entry point: ingest all data blocks into the store (V2_PLAN §8, Phase 1).

    python -m cryptorisk.study.run_ingest
"""

from __future__ import annotations

from cryptorisk.config import load_config


def main() -> None:
    _cfg = load_config()
    raise NotImplementedError("study.run_ingest - Phase 1")


if __name__ == "__main__":
    main()
