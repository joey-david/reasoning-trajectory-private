# SMT Verifier

Purpose: run small Z3 solver snippets for SAT/SMT feedback.
Inputs: Python snippet using `z3` and provided `solver`.
Outputs: SAT/UNSAT/unknown status.
CLI: `rt verify smt --input examples/smt_sat.py`
Python API: `reasoning_trajectory.verifiers.smt.SMTVerifier`.
Example: `rt verify smt --input examples/smt_sat.py`
Notes: useful for symbolic transition labels and dead-end checks.
Failure modes: requires `z3-solver`; malformed snippets raise Python errors.
