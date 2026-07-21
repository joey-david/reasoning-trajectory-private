from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.orchestration.jobs import load_job
from src.orchestration.jobs.causal_patching import pending_tasks
from src.orchestration.jobs.gold_answer_capture import (
    pending_tasks as pending_gold_tasks,
)
from src.orchestration.jobs.state_handoff_patch import (
    pending_tasks as pending_handoff_tasks,
)
from src.orchestration.jobs.state_abstraction_interchange import (
    pending_tasks as pending_abstraction_interchange_tasks,
)
from src.orchestration.jobs.state_abstraction_capture import (
    pending_tasks as pending_abstraction_capture_tasks,
)
from src.orchestration.jobs.state_transfer_capture import (
    pending_tasks as pending_transfer_capture_tasks,
)
from src.orchestration.remote import parse_workers, worker_command


class OrchestrationJobTests(unittest.TestCase):
    def test_job_loader_accepts_both_name_styles(self):
        self.assertIs(load_job("causal-patching"), load_job("causal_patching"))
        self.assertIs(
            load_job("gold-answer-capture"),
            load_job("gold_answer_capture"),
        )
        self.assertIs(
            load_job("solution-object-labeling"),
            load_job("solution_object_labeling"),
        )
        self.assertIs(
            load_job("solution-object-labeling-smoke"),
            load_job("solution_object_labeling_smoke"),
        )
        self.assertIs(
            load_job("depth-relief-calibration"),
            load_job("depth_relief_calibration"),
        )
        self.assertIs(
            load_job("state-materialization"),
            load_job("state_materialization"),
        )
        self.assertIs(load_job("state-routing"), load_job("state_routing"))
        self.assertIs(
            load_job("state-transfer-capture"),
            load_job("state_transfer_capture"),
        )
        self.assertIs(
            load_job("state-transfer-patch"),
            load_job("state_transfer_patch"),
        )
        self.assertIs(
            load_job("state-handoff-patch"),
            load_job("state_handoff_patch"),
        )
        self.assertIs(
            load_job("state-abstraction-capture"),
            load_job("state_abstraction_capture"),
        )
        self.assertIs(
            load_job("state-abstraction-interchange"),
            load_job("state_abstraction_interchange"),
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
        self.assertIn("ORCHESTRATOR_GPU_COUNT=1", command[2])
        self.assertIn("--job causal_patching", command[2])

    def test_grouped_devices_create_one_multi_gpu_worker(self):
        workers = parse_workers(
            ["upnquick", "coktailjet"],
            ["0+1", "0,1"],
        )

        self.assertEqual(
            workers,
            [
                ("upnquick", (0, 1)),
                ("coktailjet", (0,)),
                ("coktailjet", (1,)),
            ],
        )
        command = worker_command(
            "local",
            (0, 1),
            Path("runs/model/job"),
            Path("/repo"),
            "solution_object_labeling",
        )
        self.assertIn("CUDA_VISIBLE_DEVICES=0,1", command[2])
        self.assertIn("ORCHESTRATOR_GPU_COUNT=2", command[2])

    def test_worker_command_forwards_hugging_face_cache_override(self):
        with patch.dict(os.environ, {"HF_LOCAL_CACHE": "/large disk/hf"}):
            command = worker_command(
                "upnquick",
                (0, 1),
                Path("runs/model/job"),
                Path("/repo"),
                "depth_relief_qualification",
            )

        self.assertIn("export HF_LOCAL_CACHE='/large disk/hf'", command[-1])

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

    def test_state_handoff_tasks_are_heldout_and_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            run_path = Path(directory)
            (run_path / "dataset.jsonl").write_text(
                "\n".join(
                    json.dumps({"id": case_id}) for case_id in ("train", "a", "b")
                )
                + "\n",
                encoding="utf-8",
            )
            split = run_path / "depth_relief" / "state_transfer" / "split.json"
            split.parent.mkdir(parents=True)
            split.write_text(
                json.dumps(
                    {
                        "train": ["train"],
                        "validation": [],
                        "test": ["a", "b"],
                    }
                ),
                encoding="utf-8",
            )
            output = split.parent / "handoff_patches.jsonl"
            output.write_text(
                json.dumps({"id": "a", "layer": 7}) + "\n",
                encoding="utf-8",
            )
            tasks, total, complete = pending_handoff_tasks(run_path)

        self.assertEqual(total, 2)
        self.assertEqual(complete, 1)
        self.assertEqual(tasks, [{"case_index": 2}])

    def test_state_transfer_capture_requeues_incomplete_activation_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            run_path = Path(directory)
            cases = [
                {"id": "stale", "history_steps": 1},
                {"id": "complete", "history_steps": 2},
            ]
            (run_path / "dataset.jsonl").write_text(
                "".join(json.dumps(case) + "\n" for case in cases),
                encoding="utf-8",
            )
            output = run_path / "depth_relief" / "state_transfer"
            output.mkdir(parents=True)
            (output / "captures.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": "stale"}),
                        json.dumps(
                            {
                                "id": "complete",
                                "compose_positions": [
                                    {"name": "start"},
                                    {"name": "history_step_1"},
                                    {"name": "history_step_2"},
                                    {"name": "final_rule"},
                                    {"name": "answer"},
                                ],
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            activations = output / "activations"
            activations.mkdir()
            np.savez_compressed(
                activations / "stale.npz",
                compose=np.zeros(1),
                materialized=np.zeros(1),
                counterfactual=np.zeros(1),
            )
            np.savez_compressed(
                activations / "complete.npz",
                compose=np.zeros(1),
                materialized=np.zeros(1),
                counterfactual=np.zeros(1),
                compose_trace=np.zeros(1),
            )

            tasks, total, complete = pending_transfer_capture_tasks(run_path)
            retained = [
                json.loads(line)
                for line in (output / "captures.jsonl").read_text().splitlines()
            ]

        self.assertEqual(total, 2)
        self.assertEqual(complete, 1)
        self.assertEqual(tasks, [{"case_index": 0}])
        self.assertEqual([row["id"] for row in retained], ["complete"])

    def test_state_abstraction_interchange_resumes_by_pair_and_layer(self):
        with tempfile.TemporaryDirectory() as directory:
            run_path = Path(directory)
            (run_path / "config.yaml").write_text(
                json.dumps({"state_abstraction": {"causal_layers": [15, 31]}}),
                encoding="utf-8",
            )
            output = run_path / "depth_relief" / "state_abstraction"
            output.mkdir(parents=True)
            (output / "pairs.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": "a", "split": "validation"}),
                        json.dumps({"id": "b", "split": "test"}),
                        json.dumps({"id": "train", "split": "train"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (output / "interchange.jsonl").write_text(
                json.dumps({"id": "a", "layer": 15}) + "\n",
                encoding="utf-8",
            )

            tasks, total, complete = pending_abstraction_interchange_tasks(run_path)

        self.assertEqual(total, 4)
        self.assertEqual(complete, 1)
        self.assertEqual(
            tasks,
            [
                {"pair_index": 0, "layer": 31},
                {"pair_index": 1, "layer": 15},
                {"pair_index": 1, "layer": 31},
            ],
        )

    def test_state_abstraction_capture_requeues_wrong_activation_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            run_path = Path(directory)
            cases = [
                {"id": "stale", "history_steps": 2},
                {"id": "complete", "history_steps": 2},
            ]
            (run_path / "dataset.jsonl").write_text(
                "\n".join(json.dumps(case) for case in cases) + "\n",
                encoding="utf-8",
            )
            output = run_path / "depth_relief"
            activation_dir = output / "state_abstraction" / "activations"
            activation_dir.mkdir(parents=True)
            positions = [
                {"name": name}
                for name in (
                    "start",
                    "history_step_1",
                    "history_step_2",
                    "final_rule",
                    "answer",
                )
            ]
            (output / "factorization_cases.jsonl").write_text(
                "\n".join(
                    json.dumps({"id": case["id"], "compose_positions": positions})
                    for case in cases
                )
                + "\n",
                encoding="utf-8",
            )
            for case_id, compose_count in (("stale", 1), ("complete", 5)):
                np.savez_compressed(
                    activation_dir / f"{case_id}.npz",
                    compose_trace=np.zeros((compose_count, 3, 4)),
                    synthesize_trace=np.zeros((1, 3, 4)),
                    update_trace=np.zeros((2, 3, 4)),
                )

            tasks, total, complete = pending_abstraction_capture_tasks(run_path)
            retained = [
                json.loads(line)
                for line in (output / "factorization_cases.jsonl")
                .read_text()
                .splitlines()
            ]

        self.assertEqual((total, complete), (2, 1))
        self.assertEqual(tasks, [{"case_index": 0}])
        self.assertEqual([row["id"] for row in retained], ["complete"])


if __name__ == "__main__":
    unittest.main()
