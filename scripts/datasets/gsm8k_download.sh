#!/usr/bin/env bash
set -euo pipefail # This is so the script stops clearly when a command fails.

mkdir -p datasets/raw/gsm8k

curl -fL \
  https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl \
  -o datasets/raw/gsm8k/train.jsonl

curl -fL \
  https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl \
  -o datasets/raw/gsm8k/test.jsonl

echo "Got $(wc -l <datasets/raw/gsm8k/train.jsonl) lines for datasets/raw/gsm8k/train.jsonl"
echo "Got $(wc -l <datasets/raw/gsm8k/test.jsonl) lines for datasets/raw/gsm8k/test.jsonl"

mkdir -p datasets/gsm8k

python3 - <<'PY'
import json
import re
from pathlib import Path

raw_dir = Path("datasets/raw/gsm8k")
out_path = Path("datasets/gsm8k/full.jsonl")

def final_answer(answer: str) -> str:
    # GSM8K stores the final answer after "####".
    # Example: "... #### 72"
    if "####" in answer:
        value = answer.split("####")[-1].strip()
    else:
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", answer)
        value = nums[-1] if nums else ""
    return value.replace(",", "")

with out_path.open("w", encoding="utf-8") as out:
    for split in ["train", "test"]:
        source = raw_dir / f"{split}.jsonl"
        with source.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                ex = json.loads(line)

                q = ex["question"].strip()
                a = ex["answer"].strip()
                expected = final_answer(a)

                prompt = (
                    "Solve the following GSM8K problem step by step. "
                    "At the end, put only the final numeric answer in \\\\boxed{}.\n\n"
                    f"Problem:\n{q}"
                )

                row = {
                    "id": f"{split}_{i:05d}",
                    "prompt": prompt,
                    "expected_answer": expected,
                    "split": split,
                    "question": q,
                    "original_answer": a,
                }

                out.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Wrote {out_path}")
PY
