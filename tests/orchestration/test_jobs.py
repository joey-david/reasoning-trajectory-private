from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.orchestration.jobs import load_job
from src.orchestration.jobs.causal_patching import pending_tasks
from src.orchestration.remote import worker_command


class OrchestrationJobTests(unittest.TestCase):
    def test_job_loader_accepts_both_name_styles(self):
        self.assertIs(load_job("causal-patching"), load_job("causal_patching"))

    def test_local_worker_command_avoids_ssh_and_isolates_gpu(self):
        command = worker_command(
            "local",
            3,
            Path("runs/model/job"),
            Path("/repo"),
            "causal_patching",
        )
        self.assertEqual(command[:2], ["bash", "-lc"])
        self.assertIn("CUDA_VISIBLE_DEVICES=3", command[2])
        self.assertIn("--job causal_patching", command[2])

    def test_causal_patching_tasks_resume_by_full_cell_key(self):
        with tempfile.TemporaryDirectory() as directory:
            run_path = Path(directory)
            pairs_path = run_path / "pairs.jsonl"
            pairs_path.write_text(
                json.dumps({"pair_id": 7}) + "\n",
                encoding="utf-8",
            )
            (run_path / "config.yaml").write_text(
                json.dumps(
                    {
                        "patching": {
                            "pairs": str(pairs_path),
                            "patch_modes": ["full", "subspace"],
                            "conditions": [
                                "baseline",
                                "equivalent",
                                "position_random",
                                "mismatched",
                            ],
                            "continuations_per_condition": 2,
                            "max_pairs": 1,
                        }
                    }
                ),
                encoding="utf-8",
            )
            output = run_path / "patching" / "continuations.jsonl"
            output.parent.mkdir()
            output.write_text(
                json.dumps(
                    {
                        "pair_id": 7,
                        "patch_mode": "full",
                        "condition": "baseline",
                        "continuation": 0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            tasks, total, complete = pending_tasks(run_path)

        self.assertEqual(total, 16)
        self.assertEqual(complete, 1)
        self.assertEqual(len(tasks), 15)
        self.assertNotIn(
            {
                "pair_index": 0,
                "patch_mode": "full",
                "condition": "baseline",
                "continuation": 0,
            },
            tasks,
        )


if __name__ == "__main__":
    unittest.main()
