from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from reasoning_trajectory.core.registry import tool
from .base import VerificationResult


class PythonTestVerifier:
    name = "python-tests"

    def verify(self, text: str, tests: str = "", timeout: float = 5.0, **_) -> VerificationResult:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "candidate.py"
            test_path = Path(td) / "test_candidate.py"
            path.write_text(text, encoding="utf-8")
            test_path.write_text("from candidate import *\n" + tests, encoding="utf-8")
            proc = subprocess.run(["python3", str(test_path)], cwd=td, text=True, capture_output=True, timeout=timeout)
        valid = proc.returncode == 0
        return VerificationResult(
            status="valid" if valid else "invalid",
            valid=valid,
            labels=["unit_tests_passed" if valid else "unit_tests_failed"],
            message=(proc.stdout + proc.stderr).strip(),
            details={"returncode": proc.returncode},
        )


@tool(
    "python-verifier",
    "verifiers",
    "Run generated Python programs against lightweight unit tests.",
    "rt verify python --input candidate.py --tests tests.py",
    "reasoning_trajectory.verifiers.python_tests.PythonTestVerifier",
    "toolkit/docs/tools/python-verifier.md",
)
def verify_python_file(input_path: str | Path, tests_path: str | Path) -> VerificationResult:
    return PythonTestVerifier().verify(Path(input_path).read_text(), Path(tests_path).read_text())
