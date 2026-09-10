"""The study config loads and validates."""

import textwrap

import pytest

from cryptorisk.config import load_config


def test_real_config_loads():
    cfg = load_config()
    assert cfg["assets"] == ["BTC", "ETH"]
    assert cfg["seed"] == 20260101
    assert all(0.0 < a < 0.5 for a in cfg["alphas"])
    assert cfg["sample"]["oos_start"] >= cfg["sample"]["start"]


def test_missing_key_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("assets: [BTC]\nseed: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)


def test_bad_alpha_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(
        textwrap.dedent(
            """
            seed: 1
            assets: [BTC]
            sample: {start: "2018-01-01", oos_start: "2019-01-01"}
            walk_forward: {windows: [500]}
            alphas: [0.7]
            evaluation: {test_level: 0.05}
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_config(p)
