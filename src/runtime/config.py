"""Load a run folder's YAML configuration and expose it through a mapping-compatible wrapper."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class RunConfig(Mapping[str, Any]):
    """Associate raw configuration values with the run folder they configure."""

    run_path: Path
    raw: dict[str, Any]

    @classmethod
    def from_dict(cls, run_path: str | Path, data: dict[str, Any]) -> "RunConfig":
        """Build a run configuration.

        Args:
            run_path: Run folder associated with the configuration.
            data: Raw configuration values to copy.

        Returns:
            A mapping-compatible configuration with ``_run_path`` injected.
        """
        run_path = Path(run_path)
        raw = dict(data)
        raw["_run_path"] = str(run_path)
        return cls(run_path=run_path, raw=raw)

    def __getitem__(self, key: str) -> Any:
        """Return the raw configuration value for ``key``.

        Args:
            key: Configuration key to look up.

        Returns:
            The value stored under ``key``.
        """
        return self.raw[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over configuration keys.

        Returns:
            An iterator over the keys in the raw configuration.
        """
        return iter(self.raw)

    def __len__(self) -> int:
        """Return the number of raw configuration entries.

        Returns:
            The number of keys in the configuration mapping.
        """
        return len(self.raw)

    def get(self, key: str, default: Any = None) -> Any:
        """Look up a configuration value without raising for a missing key.

        Args:
            key: Configuration key to look up.
            default: Value returned when ``key`` is absent.

        Returns:
            The stored value or ``default``.
        """
        return self.raw.get(key, default)


def load_config(run_path: str | Path) -> RunConfig:
    """Load one run folder's ``config.yaml``.

    Args:
        run_path: Run folder containing the YAML configuration.

    Returns:
        The parsed, mapping-compatible run configuration.
    """
    run_path = Path(run_path)
    config_path = run_path / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return RunConfig.from_dict(run_path, config)
