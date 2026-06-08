from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from reasoning_trajectory.core.schema import Step, VerifierState


@dataclass
class VerificationResult:
    status: str
    valid: bool
    labels: list[str] = field(default_factory=list)
    score: float | None = None
    message: str | None = None
    details: dict = field(default_factory=dict)

    def to_state(self) -> VerifierState:
        return VerifierState(self.status, self.valid, self.labels, self.score, self.message, self.details)


class Verifier(Protocol):
    name: str

    def verify(self, text: str, **kwargs) -> VerificationResult:
        ...


def label_transition(previous: VerificationResult | None, current: VerificationResult) -> list[str]:
    labels = list(current.labels)
    if current.valid:
        labels.append("valid_transition")
    else:
        labels.append("invalid_transition")
    if previous and not previous.valid and current.valid:
        labels.append("recoverable_failure")
    if previous and previous.valid and not current.valid:
        labels.append("unrecoverable_failure")
    if current.details.get("goal_delta", 0) < 0:
        labels.append("goal_reducing")
    if current.details.get("goal_delta", 0) > 0:
        labels.append("goal_expanding")
    if current.details.get("branches", 0) > 1:
        labels.append("branch_point")
    if not current.valid and current.details.get("no_progress"):
        labels.append("dead_end")
    return sorted(set(labels))
