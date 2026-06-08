# Step Parser

Purpose: segment reasoning into newline, numbered, proof tactic, and code-edit steps, then pool hidden states per step.
Inputs: generated text and optional `[tokens,layers,hidden]` activations.
Outputs: step spans or schema `Step` objects.
CLI: `rt parse-steps --input text.txt --out steps.jsonl`
Python API: `reasoning_trajectory.extract.token_steps.parse_steps`.
Example: `printf 'Step 1: set x\\nStep 2: answer' > /tmp/cot.txt && rt parse-steps --input /tmp/cot.txt --out /tmp/steps.jsonl`
Notes: pooling supports `mean`, `last`, `max`, and `attention`.
Failure modes: attention pooling requires attention weights.
