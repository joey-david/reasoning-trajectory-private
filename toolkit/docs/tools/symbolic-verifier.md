# Symbolic Verifier

Purpose: verify algebraic equivalence with SymPy.
Inputs: expression and expected expression.
Outputs: verifier status and labels.
CLI: `rt verify symbolic --expr '2+2' --expected '4'`
Python API: `reasoning_trajectory.verifiers.symbolic_math.SymbolicMathVerifier`.
Example: `rt verify symbolic --expr 'sin(x)**2+cos(x)**2' --expected '1'`
Notes: expressions are parsed by SymPy.
Failure modes: missing SymPy or unparsable expressions.
