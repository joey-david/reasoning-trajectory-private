from __future__ import annotations

import numpy as np

from reasoning_trajectory.metrics.geometry import _torsion, trajectory_geometry
from reasoning_trajectory.core.schema import Step, Trajectory


def curve_traj(points):
    steps = [Step(f"s{i}", i, i + 1, str(p), {"0": list(map(float, p))}) for i, p in enumerate(points)]
    return Trajectory("curve", "curve", "synthetic", "none", "prompt", steps=steps)


def main() -> None:
    line = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    row = trajectory_geometry(curve_traj(line))
    assert abs(row["path_length"] - 2.0) < 1e-9
    assert abs(row["endpoint_distance"] - 2.0) < 1e-9
    assert row["mean_curvature"] == 0.0
    assert _torsion(line) == 0.0
    print("synthetic curves ok")


if __name__ == "__main__":
    main()
