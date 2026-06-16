from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class RunConfig(Mapping[str, Any]):
    """Thin run-config wrapper."""

    run_path: Path
    raw: dict[str, Any]

    @classmethod
    def from_dict(cls, run_path: str | Path, data: dict[str, Any]) -> "RunConfig":
        run_path = Path(run_path)
        raw = dict(data)
        raw["_run_path"] = str(run_path)
        return cls(run_path=run_path, raw=raw)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.raw)

    def __len__(self) -> int:
        return len(self.raw)

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.raw)


def load_config(run_path: str | Path) -> RunConfig:
    run_path = Path(run_path)
    config_path = run_path / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return RunConfig.from_dict(run_path, config)


def resolve_repo_path(path_text: str | Path) -> Path:
    root = Path(__file__).resolve().parents[1]
    path = Path(path_text)
    return path if path.is_absolute() else root / path
