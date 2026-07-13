from __future__ import annotations

from src.runtime.config import load_config


def test_load_config_accepts_bare_layer_slice(tmp_path) -> None:
    run_path = tmp_path / "run"
    run_path.mkdir()
    (run_path / "config.yaml").write_text(
        """
model:
  name: dummy
capture:
  enabled: true
  layers: [:] # capture all layers
""",
        encoding="utf-8",
    )

    config = load_config(run_path)

    assert config["capture"]["layers"] == "[:]"
