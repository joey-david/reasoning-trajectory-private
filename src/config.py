from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class RunConfig(Mapping[str, Any]):
    """Typed run config with raw-key compatibility for existing helpers."""

    run_path: Path
    model_name: str
    dataset_path: str
    backend: str = "hf"
    device_map: str | dict[str, Any] | None = "auto"
    torch_dtype: str | None = None
    trust_remote_code: bool = False
    max_new_tokens: int = 1024
    seeds: list[int] = field(default_factory=lambda: [0])
    temperatures: list[float] = field(default_factory=lambda: [0.0])
    layers: list[int] = field(default_factory=lambda: [-1])
    activation_storage_dtype: str = "float16"
    capture_diagnostics: bool = True
    top_p: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, run_path: str | Path, data: dict[str, Any]) -> "RunConfig":
        run_path = Path(run_path)
        raw = dict(data)
        raw["_run_path"] = str(run_path)

        return cls(
            run_path=run_path,
            model_name=str(required(raw, "model_name")),
            dataset_path=str(required(raw, "dataset_path")),
            backend=str(raw.get("backend", "hf")),
            device_map=raw.get("device_map", "auto"),
            torch_dtype=raw.get("torch_dtype"),
            trust_remote_code=bool(raw.get("trust_remote_code", False)),
            max_new_tokens=int(raw.get("max_new_tokens", 1024)),
            seeds=[int(seed) for seed in raw.get("seeds", [0])],
            temperatures=[float(temp) for temp in raw.get("temperatures", [0.0])],
            layers=[int(layer) for layer in raw.get("layers", [-1])],
            activation_storage_dtype=str(
                raw.get("activation_storage_dtype", "float16")
            ),
            capture_diagnostics=bool(raw.get("capture_diagnostics", True)),
            top_p=None if raw.get("top_p") is None else float(raw["top_p"]),
            raw=raw,
        )

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.raw)
        data.update(
            {
                "_run_path": str(self.run_path),
                "model_name": self.model_name,
                "dataset_path": self.dataset_path,
                "backend": self.backend,
                "device_map": self.device_map,
                "torch_dtype": self.torch_dtype,
                "trust_remote_code": self.trust_remote_code,
                "max_new_tokens": self.max_new_tokens,
                "seeds": self.seeds,
                "temperatures": self.temperatures,
                "layers": self.layers,
                "activation_storage_dtype": self.activation_storage_dtype,
                "capture_diagnostics": self.capture_diagnostics,
                "top_p": self.top_p,
            }
        )
        return data


def load_config(run_path: str | Path) -> RunConfig:
    """Load `config.yaml` from a run folder.

    Implementing notes:
    - Returns a `RunConfig`, while preserving mapping-style access for callers
      that still expect a config dict.
    """
    run_path = Path(run_path)
    config_path = run_path / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return RunConfig.from_dict(run_path, config)


def resolve_repo_path(path_text: str | Path) -> Path:
    """Resolve a path relative to the repository root.

    Useful Python tools:
    - `Path(__file__).resolve().parents[1]` finds this repo root from `src/`.
    - `Path.is_absolute()` tells you whether a path already starts at `/`.
    """
    root = Path(__file__).resolve().parents[1]
    path = Path(path_text)
    return path if path.is_absolute() else root / path


def required(config: dict[str, Any], key: str) -> Any:
    value = config.get(key)
    if value is None:
        raise KeyError(f"Missing required config key: {key}")
    return value
