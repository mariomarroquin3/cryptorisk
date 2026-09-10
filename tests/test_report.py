"""Phase 6: the report assembler produces the expected files from committed
pipeline output. Artefact-gated — skipped without the store + the eval CSVs.
"""

from __future__ import annotations

import pytest

from cryptorisk.config import load_config, repo_root

_RES = repo_root() / load_config()["paths"]["results"]
_NEED = ["backtests.parquet", "eval_fz0_mcs.csv", "eval_coverage.csv"]
pytestmark = pytest.mark.skipif(
    not (repo_root() / load_config()["paths"]["store"]).exists()
    or not all((_RES / n).exists() for n in _NEED),
    reason="run the pipeline through `make evaluate` first",
)


def test_report_builds_results_and_cards(tmp_path, monkeypatch):
    from cryptorisk.models.registry import all_models
    from cryptorisk.study import report

    D = report._load()

    figs = report.all_figures(D)
    assert figs["fz0_mcs"] and figs["returns_rv"]
    for name in figs.values():
        if name:
            assert (repo_root() / "docs" / "figures" / name).exists()

    md = report.results_md(D, figs)
    for anchor in (
        "# cryptorisk: empirical results",
        "## 3. Headline",
        "Model Confidence Set",
        "## 8. Decision layer",
        "## 9. What the numbers do not settle",
    ):
        assert anchor in md
    assert "nan |" not in md  # no stray NaN cells in the tables

    cards = report.model_cards(D)
    assert len(cards) == len(all_models())
    for m in all_models():
        card = (repo_root() / "docs" / "model_cards" / f"{report._slug(m.name)}.md").read_text(
            encoding="utf-8"
        )
        assert f"# Model card: {m.name}" in card
        assert "Out-of-sample scorecard" in card


def test_report_main_runs():
    from cryptorisk.study import report

    report.main()  # writes docs/results.md, docs/model_cards/, docs/figures/
    assert (repo_root() / "docs" / "results.md").exists()
