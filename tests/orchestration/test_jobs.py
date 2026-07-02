from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.orchestration.jobs import load_job
from src.orchestration.jobs.boundary_intervention import (
    pending_tasks as pending_boundary_tasks,
)
from src.orchestration.jobs.causal_patching import pending_tasks
from src.orchestration.jobs.gold_answer_capture import (
    pending_tasks as pending_gold_tasks,
)
from src.orchestration.remote import worker_command


class OrchestrationJobTests(unittest.TestCase):
    def test_job_loader_accepts_both_name_styles(self):
        self.assertIs(load_job("causal-patching"), load_job("causal_patching"))
        self.assertIs(
            load_job("gold-answer-capture"),
            load_job("gold_answer_capture"),
        )

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

    def test_boundary_tasks_resume_by_point_condition_and_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            run_path = Path(directory)
            manifest = run_path / "boundaries.jsonl"
            manifest.write_text(
                json.dumps({"point_id": 4}) + "\n",
                encoding="utf-8",
            )
            (run_path / "config.yaml").write_text(
                json.dumps(
                    {
                        "boundary_intervention": {
                            "manifest": str(manifest),
                            "conditions": ["baseline", "zero"],
                            "continuations_per_condition": 2,
                        }
                    }
                ),
                encoding="utf-8",
            )
            output = run_path / "interventions" / "continuations.jsonl"
            output.parent.mkdir()
            output.write_text(
                json.dumps(
                    {
                        "point_id": 4,
                        "condition": "zero",
                        "continuation": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            tasks, total, complete = pending_boundary_tasks(run_path)

        self.assertEqual(total, 4)
        self.assertEqual(complete, 1)
        self.assertEqual(len(tasks), 3)

    def test_gold_answer_tasks_resume_by_sample_id(self):
        with tempfile.TemporaryDirectory() as directory:
            run_path = Path(directory)
            dataset = run_path / "dataset.jsonl"
            dataset.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "a",
                                "question": "A?",
                                "gold_answer": "A.",
                            }
                        ),
                        json.dumps(
                            {
                                "id": "b",
                                "question": "B?",
                                "gold_answer": "B.",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (run_path / "config.yaml").write_text(
                json.dumps(
                    {
                        "dataset": {
                            "source": "jsonl",
                            "path": str(dataset),
                            "adapter": "plain_question",
                        }
                    }
                ),
                encoding="utf-8",
            )
            manifest = run_path / "gold_answers" / "manifest.jsonl"
            manifest.parent.mkdir()
            manifest.write_text(
                json.dumps(
                    {
                        "sample_id": "a",
                        "hidden_states_file": "gold_answers/a.npz",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            tasks, total, complete = pending_gold_tasks(run_path)

        self.assertEqual(total, 2)
        self.assertEqual(complete, 1)
        self.assertEqual(tasks, [{"sample_index": 1}])


if __name__ == "__main__":
    unittest.main()
