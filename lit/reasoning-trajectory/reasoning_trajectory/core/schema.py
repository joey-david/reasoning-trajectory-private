from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class VerifierState:
    """Process-level verifier feedback for one reasoning step."""

    status: str = "unknown"
    valid_transition: bool | None = None
    labels: list[str] = field(default_factory=list)
    score: float | None = None
    message: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SolutionObject:
    """External solution object aligned to hidden reasoning."""

    object_id: str
    object_type: str
    text: str = ""
    ast: dict[str, Any] | None = None
    states: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Step:
    """One segmented reasoning step and optional model artifacts."""

    step_id: str
    token_start: int
    token_end: int
    text: str
    hidden_states: dict[str, list[float] | list[list[float]]] = field(default_factory=dict)
    logits_optional: list[float] | None = None
    verifier_state_optional: VerifierState | None = None
    labels: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """Shared trajectory schema consumed by all tools."""

    trajectory_id: str
    problem_id: str
    dataset: str
    model_name: str
    prompt: str
    seed: int = 0
    temperature: float = 0.0
    decoding_method: str = "greedy"
    final_text: str = ""
    final_answer: str | None = None
    final_correct: bool | None = None
    solution_object_id: str | None = None
    steps: list[Step] = field(default_factory=list)
    solution_object: SolutionObject | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.metadata.setdefault("created_at", _now())
        self.metadata.setdefault("repo_commit", "unknown")
        self.metadata.setdefault("config_hash", "unknown")

    def validate(self) -> list[str]:
        errors: list[str] = []
        required = ["trajectory_id", "problem_id", "dataset", "model_name", "prompt"]
        for name in required:
            if not getattr(self, name):
                errors.append(f"missing {name}")
        last_end = -1
        seen = set()
        for step in self.steps:
            if step.step_id in seen:
                errors.append(f"duplicate step_id {step.step_id}")
            seen.add(step.step_id)
            if step.token_start < 0 or step.token_end < step.token_start:
                errors.append(f"invalid token span for {step.step_id}")
            if step.token_start < last_end:
                errors.append(f"overlapping step span at {step.step_id}")
            last_end = max(last_end, step.token_end)
        return errors

    def require_valid(self) -> None:
        errors = self.validate()
        if errors:
            raise ValueError("; ".join(errors))


def to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_dict(v) for v in obj]
    return obj


def verifier_from_dict(data: dict[str, Any] | None) -> VerifierState | None:
    return None if data is None else VerifierState(**data)


def solution_from_dict(data: dict[str, Any] | None) -> SolutionObject | None:
    return None if data is None else SolutionObject(**data)


def step_from_dict(data: dict[str, Any]) -> Step:
    data = dict(data)
    data["verifier_state_optional"] = verifier_from_dict(data.get("verifier_state_optional"))
    return Step(**data)


def trajectory_from_dict(data: dict[str, Any]) -> Trajectory:
    data = dict(data)
    data["steps"] = [step_from_dict(step) for step in data.get("steps", [])]
    data["solution_object"] = solution_from_dict(data.get("solution_object"))
    return Trajectory(**data)
