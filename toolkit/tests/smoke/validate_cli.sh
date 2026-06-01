#!/usr/bin/env bash
set -euo pipefail
cat >/tmp/rt_validate_extract.yaml <<'YAML'
dataset: validate
model_name: mock
seeds: [0, 1]
temperatures: [0.0]
mock_layers: 3
mock_hidden: 8
layers: [0, 1, 2]
prompts:
  - problem_id: validate
    expected_answer: 4
    prompt: What is 2 squared?
YAML
rt extract --config /tmp/rt_validate_extract.yaml --out /tmp/rt_validate
rt metrics --input /tmp/rt_validate --out /tmp/rt_validate/metrics
rt compression --input /tmp/rt_validate --out /tmp/rt_validate/compression.jsonl --dims 2
rt basins --input /tmp/rt_validate --out /tmp/rt_validate/basins.json --clusters 2
rt plot --input /tmp/rt_validate --out /tmp/rt_validate/traj.html
rt dashboard --input /tmp/rt_validate --out /tmp/rt_validate/dashboard.html
rt report --input /tmp/rt_validate --out /tmp/rt_validate/report.md
cat >/tmp/rt_candidate.py <<'PY'
def solve():
    return 4
PY
cat >/tmp/rt_test_candidate.py <<'PY'
assert solve() == 4
PY
python3 - <<'PY'
import numpy as np
np.save('/tmp/rt_hidden.npy', np.arange(24, dtype=float).reshape(3, 8))
np.save('/tmp/rt_vector.npy', np.ones(8))
np.save('/tmp/rt_unembed.npy', np.ones((5, 8)))
PY
rt verify python --input /tmp/rt_candidate.py --tests /tmp/rt_test_candidate.py
rt verify symbolic --expr '2+2' --expected '4'
rt list-tools
