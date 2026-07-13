"""Read run configuration files for reusable trajectory tools."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import yaml


def load_run_config(run_path: str | Path) -> dict[str, Any]:
    """Load a run folder's ``config.yaml`` with repository YAML extensions."""
    config_path = Path(run_path) / "config.yaml"
    text = _quote_bare_layer_slice(config_path.read_text(encoding="utf-8"))
    return yaml.safe_load(text) or {}


def _quote_bare_layer_slice(text: str) -> str:
    """Allow ``layers: [:]`` as a compact all-layers capture sentinel."""
    return re.sub(
        r"(^\s*layers\s*:\s*)\[:\](\s*(?:#.*)?$)",
        r"\1'[:]'\2",
        text,
        flags=re.MULTILINE,
    )
