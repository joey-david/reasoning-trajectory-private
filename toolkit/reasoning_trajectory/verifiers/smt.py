from __future__ import annotations

from .base import VerificationResult
from pathlib import Path

from reasoning_trajectory.core.registry import tool


class SMTVerifier:
    name = "smt"

    def verify(self, text: str, **_) -> VerificationResult:
        try:
            import z3
        except ImportError as exc:
            raise ImportError("SMT verifier requires z3-solver") from exc
        solver = z3.Solver()
        env = {"z3": z3, "solver": solver}
        exec(text, env, env)
        result = solver.check()
        ok = result == z3.sat
        return VerificationResult("sat" if ok else str(result), ok, ["smt_sat" if ok else "smt_unsat_or_unknown"], details={"result": str(result)})


@tool(
    "smt-verifier",
    "verifiers",
    "Run Z3-backed SMT checks from small Python snippets.",
    "rt verify smt --input examples/smt_sat.py",
    "reasoning_trajectory.verifiers.smt.SMTVerifier",
    "toolkit/docs/tools/smt-verifier.md",
)
def verify_smt_file(input_path: str | Path) -> VerificationResult:
    return SMTVerifier().verify(Path(input_path).read_text(encoding="utf-8"))
