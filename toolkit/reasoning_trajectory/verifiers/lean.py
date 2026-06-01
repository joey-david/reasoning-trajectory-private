from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from reasoning_trajectory.core.registry import tool
from .base import VerificationResult


class LeanVerifier:
    name = "lean"

    def verify(self, text: str, timeout: float = 10.0, **_) -> VerificationResult:
        lean = shutil.which("lean")
        if not lean:
            return VerificationResult("missing_dependency", False, ["lean_missing"], message="lean executable not found")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "Candidate.lean"
            path.write_text(text, encoding="utf-8")
            try:
                proc = subprocess.run([lean, str(path)], text=True, capture_output=True, timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                return VerificationResult("timeout", False, ["lean_timeout"], message=str(exc))
        ok = proc.returncode == 0
        return VerificationResult("valid" if ok else "invalid", ok, ["lean_checked" if ok else "lean_error"], message=(proc.stdout + proc.stderr).strip())


@tool(
    "lean-verifier",
    "verifiers",
    "Check Lean files with the local Lean executable.",
    "rt verify lean --input examples/lean_ok.lean",
    "reasoning_trajectory.verifiers.lean.LeanVerifier",
    "toolkit/docs/tools/lean-verifier.md",
)
def verify_lean_file(input_path: str | Path) -> VerificationResult:
    return LeanVerifier().verify(Path(input_path).read_text(encoding="utf-8"))
