from __future__ import annotations

from reasoning_trajectory.core.registry import tool
from .base import VerificationResult


class SymbolicMathVerifier:
    name = "symbolic-math"

    def verify(self, text: str, expected: str | None = None, **_) -> VerificationResult:
        try:
            import sympy as sp
        except ImportError as exc:
            raise ImportError("symbolic verifier requires sympy") from exc
        if expected is None:
            return VerificationResult("unknown", False, ["missing_expected"], message="expected expression is required")
        lhs, rhs = sp.sympify(text), sp.sympify(expected)
        ok = bool(sp.simplify(lhs - rhs) == 0)
        return VerificationResult("valid" if ok else "invalid", ok, ["symbolic_equal" if ok else "symbolic_mismatch"])


@tool(
    "symbolic-verifier",
    "verifiers",
    "Verify algebraic equivalence with SymPy.",
    "rt verify symbolic --expr '2+2' --expected '4'",
    "reasoning_trajectory.verifiers.symbolic_math.SymbolicMathVerifier",
    "toolkit/docs/tools/symbolic-verifier.md",
)
def verify_symbolic(expr: str, expected: str) -> VerificationResult:
    return SymbolicMathVerifier().verify(expr, expected)
