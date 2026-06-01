from __future__ import annotations

from .schema import SolutionObject, Step, Trajectory, VerifierState


def tiny_trajectory() -> Trajectory:
    steps = [
        Step("s1", 0, 4, "Step 1: set x=2", {"0": [0.0, 0.0, 0.0], "1": [0.0, 0.1, 0.0]}, labels=["setup"]),
        Step("s2", 4, 8, "Step 2: square x", {"0": [1.0, 0.0, 0.0], "1": [0.9, 0.2, 0.1]}, labels=["compute"]),
        Step(
            "s3",
            8,
            12,
            "Step 3: answer 4",
            {"0": [2.0, 0.0, 0.0], "1": [1.8, 0.2, 0.0]},
            verifier_state_optional=VerifierState(status="valid", valid_transition=True, labels=["goal_reducing"]),
            labels=["answer"],
        ),
    ]
    return Trajectory(
        trajectory_id="tiny-0",
        problem_id="square-two",
        dataset="fixture",
        model_name="mock-linear",
        prompt="What is 2 squared?",
        final_text="\n".join(s.text for s in steps) + "\n#### 4",
        final_answer="4",
        final_correct=True,
        solution_object_id="square-two-program",
        solution_object=SolutionObject("square-two-program", "program", "def solve(): return 2 * 2"),
        steps=steps,
        metadata={"created_at": "2026-01-01T00:00:00+00:00", "repo_commit": "fixture", "config_hash": "fixture"},
    )
