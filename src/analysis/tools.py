from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.analysis.activation_norms import write_activation_norms
from src.analysis.components import write_pca_components
from src.analysis.generation_summary import write_generation_summary
from src.analysis.trajectory import write_trajectory_projection


ToolRunner = Callable[..., Path]


@dataclass(frozen=True)
class AnalysisTool:
    name: str
    label: str
    description: str
    runner: ToolRunner
    params: dict[str, Any] = field(default_factory=dict)


TOOLS: dict[str, AnalysisTool] = {
    "generation_summary": AnalysisTool(
        "generation_summary",
        "Generation Summary",
        "Token counts, output lengths, average selected-token log-probability, and answer labels.",
        write_generation_summary,
    ),
    "activation_norms": AnalysisTool(
        "activation_norms",
        "Activation Norms",
        "Layer-wise activation magnitude summary for each generation.",
        write_activation_norms,
    ),
    "trajectory_projection": AnalysisTool(
        "trajectory_projection",
        "Trajectory Projection",
        "3D PCA or t-SNE projection of layer activations sampled along each generated trajectory.",
        write_trajectory_projection,
        {"layer": None, "interval": 16, "method": "pca", "max_points": 12000, "skip_first": 2, "skip_last": 2},
    ),
    "pca_components": AnalysisTool(
        "pca_components",
        "PCA Components",
        "Amplitude and explained variance of the first principal components.",
        write_pca_components,
        {"layer": None, "n": 24, "max_vectors": 20000, "skip_first": 2, "skip_last": 2},
    ),
}


def run_tool(config: dict[str, Any], name: str, params: dict[str, Any] | None = None) -> Path:
    tool = TOOLS[name]
    kwargs = {**tool.params, **(params or {})}
    return tool.runner(config, **kwargs)


def run_all_tools(config: dict[str, Any], params: dict[str, Any] | None = None) -> list[Path]:
    outputs = []
    for name in TOOLS:
        outputs.append(run_tool(config, name, (params or {}).get(name, {})))
    return outputs


def tool_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": tool.name,
            "label": tool.label,
            "description": tool.description,
            "params": tool.params,
        }
        for tool in TOOLS.values()
    ]
