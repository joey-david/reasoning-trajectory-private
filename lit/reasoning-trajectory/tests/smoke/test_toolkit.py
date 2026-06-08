from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from reasoning_trajectory.analysis import pca_compress
from reasoning_trajectory.core.fixtures import tiny_trajectory
from reasoning_trajectory.core.storage import load_trajectories, save_jsonl
from reasoning_trajectory.extract.token_steps import aggregate_step_hidden_states, parse_steps
from reasoning_trajectory.metrics.alignment import alignment_summary
from reasoning_trajectory.metrics.geometry import trajectory_geometry
from reasoning_trajectory.verifiers.symbolic_math import SymbolicMathVerifier


def main() -> None:
    config = Path("/tmp/rt_smoke_extract.yaml")
    config.write_text(
        """
dataset: smoke
model_name: mock
seeds: [0, 1]
temperatures: [0.0]
mock_layers: 3
mock_hidden: 8
layers: [0, 1, 2]
prompts:
  - problem_id: smoke
    expected_answer: 4
    prompt: What is 2 squared?
""".strip()
        + "\n",
        encoding="utf-8",
    )
    candidate = Path("/tmp/rt_candidate.py")
    candidate.write_text("def solve():\n    return 4\n", encoding="utf-8")
    candidate_test = Path("/tmp/rt_test_candidate.py")
    candidate_test.write_text("assert solve() == 4\n", encoding="utf-8")
    lean_file = Path("/tmp/rt_lean_ok.lean")
    lean_file.write_text("example : 1 + 1 = 2 := by\n  norm_num\n", encoding="utf-8")
    smt_file = Path("/tmp/rt_smt_sat.py")
    smt_file.write_text('x = z3.Int("x")\nsolver.add(x > 0)\n', encoding="utf-8")

    run_dir = Path("/tmp/rt_smoke")
    save_jsonl([tiny_trajectory()], run_dir / "trajectories.jsonl")
    assert load_trajectories(run_dir)[0].trajectory_id == "tiny-0"

    spans = parse_steps("Step 1: setup\n2. compute\nintro x\nreturn 4")
    assert len(spans) == 4
    hidden = np.ones((12, 3, 4))
    pooled = aggregate_step_hidden_states(hidden, spans, pooling="mean")
    assert pooled and "0" in pooled[0]

    geom = trajectory_geometry(tiny_trajectory())
    assert geom["path_length"] > 0
    x = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)
    y = np.array([[0, 0], [1, 1], [2, 0]], dtype=float)
    assert alignment_summary(x, y)["dtw"] >= 0
    assert pca_compress(np.eye(4), 2)["error"] >= 0
    assert SymbolicMathVerifier().verify("2+2", "4").valid
    np.save("/tmp/rt_hidden.npy", np.arange(24, dtype=float).reshape(3, 8))
    np.save("/tmp/rt_donor.npy", np.ones((3, 8), dtype=float))
    np.save("/tmp/rt_vector.npy", np.ones((8,), dtype=float))
    np.save("/tmp/rt_unembed.npy", np.ones((5, 8), dtype=float))
    np.save("/tmp/rt_weights.npy", np.ones((3, 8), dtype=float))
    np.save("/tmp/rt_features.npy", np.eye(4))
    np.save("/tmp/rt_labels.npy", np.array([0, 1, 0, 1]))

    commands = [
        ["rt", "extract", "--config", str(config), "--out", "/tmp/rt_smoke_extract"],
        ["rt", "metrics", "--input", "/tmp/rt_smoke_extract", "--out", "/tmp/rt_smoke_extract/metrics"],
        ["rt", "compression", "--input", "/tmp/rt_smoke_extract", "--out", "/tmp/rt_smoke_extract/compression.jsonl"],
        ["rt", "plot", "--input", "/tmp/rt_smoke_extract", "--out", "/tmp/rt_smoke_extract/traj.html"],
        ["rt", "basins", "--input", "/tmp/rt_smoke_extract", "--out", "/tmp/rt_smoke_extract/basins.json", "--clusters", "2"],
        ["rt", "dashboard", "--input", "/tmp/rt_smoke_extract", "--out", "/tmp/rt_smoke_extract/dashboard.html"],
        ["rt", "report", "--input", "/tmp/rt_smoke_extract", "--out", "/tmp/rt_smoke_extract/report.md"],
        ["rt", "verify", "python", "--input", str(candidate), "--tests", str(candidate_test)],
        ["rt", "verify", "symbolic", "--expr", "2+2", "--expected", "4"],
        ["rt", "verify", "lean", "--input", str(lean_file)],
        ["rt", "verify", "smt", "--input", str(smt_file)],
        ["rt", "list-tools"],
        ["rt", "doctor"],
    ]
    for command in commands:
        result = subprocess.run(command, text=True, capture_output=True)
        assert result.returncode == 0, (command, result.stdout, result.stderr)
    tools = subprocess.check_output(["rt", "list-tools"], text=True)
    for name in ["extract", "trajectory-3d", "geometry", "alignment", "dashboard", "python-verifier", "compression", "basins", "run-report"]:
        assert name in tools
    print(json.dumps({"status": "ok", "geometry_path_length": geom["path_length"]}))


if __name__ == "__main__":
    main()
