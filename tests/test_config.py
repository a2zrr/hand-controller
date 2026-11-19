"""Config loading tests."""

from pathlib import Path

import pytest

from hand_controller.config import Config, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_missing_file_gives_defaults(tmp_path):
    assert load_config(tmp_path / "nope.toml") == Config()


def test_shipped_config_matches_defaults():
    # config.toml documents every default; if it drifts from the dataclasses, this fails.
    assert load_config(ROOT / "config.toml") == Config()


def test_values_override_defaults(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[output]\nport = 6000\n[smoothing]\ndirection_frames = 7\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.output.port == 6000 and cfg.smoothing.direction_frames == 7
    assert cfg.output.host == "127.0.0.1"


def test_unknown_key_is_an_error(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[output]\nprot = 6000\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prot"):
        load_config(p)


def test_unknown_section_is_an_error(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[nonsense]\nx = 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)
