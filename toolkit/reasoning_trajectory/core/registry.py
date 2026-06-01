from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ToolSpec:
    name: str
    category: str
    description: str
    cli: str
    api: str
    doc: str
    dashboard: bool = False


TOOLS: dict[str, ToolSpec] = {}


def register(spec: ToolSpec) -> ToolSpec:
    TOOLS[spec.name] = spec
    return spec


def tool(name: str, category: str, description: str, cli: str, api: str, doc: str, dashboard: bool = False) -> Callable:
    def wrap(fn: Callable) -> Callable:
        register(ToolSpec(name, category, description, cli, api, doc, dashboard))
        return fn
    return wrap


def list_tools() -> list[ToolSpec]:
    import reasoning_trajectory.analysis  # noqa: F401
    import reasoning_trajectory.branching  # noqa: F401
    import reasoning_trajectory.extract.generations  # noqa: F401
    import reasoning_trajectory.extract.hf_check  # noqa: F401
    import reasoning_trajectory.extract.token_steps  # noqa: F401
    import reasoning_trajectory.visualize.trajectory_3d  # noqa: F401
    import reasoning_trajectory.metrics.geometry  # noqa: F401
    import reasoning_trajectory.metrics.alignment  # noqa: F401
    import reasoning_trajectory.verifiers.python_tests  # noqa: F401
    import reasoning_trajectory.verifiers.symbolic_math  # noqa: F401
    import reasoning_trajectory.verifiers.lean  # noqa: F401
    import reasoning_trajectory.verifiers.smt  # noqa: F401
    import reasoning_trajectory.dashboard.app  # noqa: F401
    return sorted(TOOLS.values(), key=lambda s: (s.category, s.name))
