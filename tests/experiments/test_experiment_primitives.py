from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.experiments.common import robust_spike_indices
from src.experiments.symbolic import extract_symbolic_updates
from src.prompting.templates import build_prompt
from src.runtime.artifact_store import (
    load_component_states_npz,
    load_hidden_states_npz,
    save_hidden_states_npz,
)
from src.runtime.paths import REPO_ROOT, resolve_repo_path


class FakeTokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        return "|".join(
            f"{message['role']}={message['content']}" for message in messages
        )


class ExperimentPrimitiveTests(unittest.TestCase):
    def test_symbolic_updates_verify_arithmetic_and_keep_final_extract(self):
        text = (
            "First 7 + 4 = 11. A mistaken 2 + 2 = 5 is ignored. "
            "The answer is 11. Final Answer: 11"
        )
        updates = extract_symbolic_updates(text, token_count=40)
        self.assertEqual(
            [update.operator for update in updates],
            ["OPERATE", "EXTRACT"],
        )
        self.assertEqual(updates[0].operation_signature, "ADD")
        self.assertEqual(updates[-1].value, 11)

    def test_robust_spikes_are_local_and_separated(self):
        magnitudes = np.ones(20, dtype=np.float32)
        magnitudes[[5, 6, 15]] = [10, 8, 12]
        self.assertEqual(
            robust_spike_indices(magnitudes, min_distance=3).tolist(),
            [5, 15],
        )

    def test_chat_demonstrations_precede_final_question(self):
        prompt = build_prompt(
            {"question": "Target?"},
            {
                "mode": "chat",
                "system": "System",
                "instruction": "Solve.",
                "demonstrations": [
                    {"user": "Example?", "assistant": "Example answer."}
                ],
            },
            FakeTokenizer(),
        )
        self.assertEqual(
            prompt,
            "system=System|user=Example?|assistant=Example answer.|"
            "user=Solve.\n\nTarget?",
        )

    def test_component_states_round_trip_with_residuals(self):
        values = np.arange(48, dtype=np.float32).reshape(3, 2, 8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "states.npz"
            save_hidden_states_npz(
                path=path,
                hidden_states=values,
                component_states={"mlp_output": values / 2},
                layer_indices=[9, -1],
                storage_dtype="float16",
            )
            residual, residual_layers = load_hidden_states_npz(path)
            mlp, mlp_layers = load_component_states_npz(path, "mlp_output")
        np.testing.assert_allclose(residual, values, atol=0.02)
        np.testing.assert_allclose(mlp, values / 2, atol=0.02)
        self.assertEqual(residual_layers, [9, -1])
        self.assertEqual(mlp_layers, [9, -1])

    def test_relative_paths_resolve_from_checkout_root(self):
        self.assertEqual(resolve_repo_path("runs"), REPO_ROOT / "runs")
        self.assertTrue((REPO_ROOT / "src").is_dir())


if __name__ == "__main__":
    unittest.main()
