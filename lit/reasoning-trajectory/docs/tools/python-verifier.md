# Python Verifier

Purpose: run candidate Python programs against unit tests.
Inputs: candidate source and test source.
Outputs: verifier status and labels.
CLI: `rt verify python --input examples/candidate.py --tests examples/test_candidate.py`
Python API: `reasoning_trajectory.verifiers.python_tests.PythonTestVerifier`.
Example: `rt verify python --input examples/candidate.py --tests examples/test_candidate.py`
Notes: each run occurs in a temporary directory.
Failure modes: timeouts and assertion failures return invalid verifier states.
