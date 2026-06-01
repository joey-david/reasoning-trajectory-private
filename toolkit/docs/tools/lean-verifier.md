# Lean Verifier

Purpose: check Lean proof snippets with the local `lean` executable.
Inputs: `.lean` file.
Outputs: verifier status and compiler messages.
CLI: `rt verify lean --input examples/lean_ok.lean`
Python API: `reasoning_trajectory.verifiers.lean.LeanVerifier`.
Example: `rt verify lean --input examples/lean_ok.lean`
Notes: uses executable Lean, so project imports should be run from an appropriate Lake environment.
Failure modes: missing Lean executable or invalid proof returns a non-valid result.
