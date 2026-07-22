"""CPU tiny-model smoke for LoRA save, reload, resume, and evaluation."""

from __future__ import annotations

import math
import json
from pathlib import Path
import tempfile
from typing import Any

from src.runtime.config import load_config

from .state_handoff_data import TEST_PATH, TRAIN_PATH, prepare_state_handoff_datasets, read_programs
from .state_handoff_evaluation import evaluate_program_hf, evaluate_state_handoff_condition
from .state_handoff_training import read_training_metrics, train_state_handoff_condition


class _CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if add_special_tokens:
            raise ValueError("The smoke tokenizer does not add special tokens")
        return [ord(character) + 2 for character in text]

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_tensors: str | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        import torch

        ids = self.encode(text, add_special_tokens=add_special_tokens)
        if return_tensors == "pt":
            tensor = torch.tensor([ids], dtype=torch.long)
            return {"input_ids": tensor, "attention_mask": torch.ones_like(tensor)}
        return {"input_ids": ids}


def _write_smoke_config(run_path: Path) -> None:
    (run_path / "config.yaml").write_text(
        """
model:
  name: local-tiny-llama
  revision: smoke
state_handoff_training:
  dataset:
    bits: 3
    seed: 17
    train_examples: 32
    validation_examples: 32
    train_horizons: [1, 2]
    validation_horizons: [1, 2]
    test_horizons: [2, 4, 8]
    train_program_contexts: 1
    validation_program_contexts: 1
    test_program_contexts: 1
    test_paths_per_state: 2
  lora:
    rank: 2
    alpha: 4
    dropout: 0
    target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  training:
    seed: 31
    max_sequence_length: 768
    microbatch: 2
    gradient_accumulation: 1
    learning_rate: 0.001
    warmup_ratio: 0
    epochs: 2
    evaluation_interval: 1
    max_gradient_norm: 1.0
    gradient_checkpointing: false
  prompt:
    mode: plain
""".strip()
        + "\n"
    )


def run_tiny_smoke(_source_run_path: Path) -> dict[str, Any]:
    """Run two resumed CPU steps and verify adapter and evaluation contracts."""
    import torch
    from peft import PeftModel
    from transformers import LlamaConfig, LlamaForCausalLM

    repository_tmp = Path(__file__).resolve().parents[3] / ".tmp"
    repository_tmp.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="state_handoff_smoke_", dir=repository_tmp
    ) as directory:
        run_path = Path(directory)
        _write_smoke_config(run_path)
        prepare_state_handoff_datasets(run_path)
        tokenizer = _CharacterTokenizer()
        torch.manual_seed(7)
        base = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=512,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=1024,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=1,
            )
        )
        base_config_path = run_path / "tiny_base"
        base.config._name_or_path = str(base_config_path)
        base.config.save_pretrained(base_config_path)
        base_state = {
            name: value.detach().clone() for name, value in base.state_dict().items()
        }

        def fresh_base() -> Any:
            model = LlamaForCausalLM(base.config)
            model.load_state_dict(base_state)
            return model

        first = train_state_handoff_condition(
            run_path,
            "explicit_handoff",
            max_optimizer_steps=1,
            model=fresh_base(),
            tokenizer=tokenizer,
            enforce_phase1_gate=False,
        )
        second = train_state_handoff_condition(
            run_path,
            "explicit_handoff",
            max_optimizer_steps=2,
            model=fresh_base(),
            tokenizer=tokenizer,
            enforce_phase1_gate=False,
        )
        metrics = read_training_metrics(run_path, "explicit_handoff")
        if [row["optimizer_step"] for row in metrics] != [1, 2]:
            raise AssertionError("Resume duplicated or skipped optimizer metrics")
        losses = [
            row[key]
            for row in metrics
            for key in ("total_loss", "state_token_loss", "answer_loss")
        ]
        if not all(value is not None and math.isfinite(float(value)) for value in losses):
            raise AssertionError("Tiny training reported a non-finite loss")

        adapter_path = Path(second["final_adapter"])
        loaded = PeftModel.from_pretrained(fresh_base(), adapter_path).eval()
        case = read_programs(run_path / TEST_PATH)[0]
        prompt = load_config(run_path).get("state_handoff_training", {}).get(
            "prompt", {}
        )
        before = evaluate_program_hf(
            model=loaded,
            tokenizer=tokenizer,
            case=case,
            prompt_config=prompt,
            condition="explicit_handoff",
        )
        roundtrip = run_path / "roundtrip_adapter"
        loaded.save_pretrained(roundtrip, safe_serialization=True)
        reloaded = PeftModel.from_pretrained(fresh_base(), roundtrip).eval()
        after = evaluate_program_hf(
            model=reloaded,
            tokenizer=tokenizer,
            case=case,
            prompt_config=prompt,
            condition="explicit_handoff",
        )
        predictions_before = {
            name: value["unconstrained_prediction"]
            for name, value in before["conditions"].items()
        }
        predictions_after = {
            name: value["unconstrained_prediction"]
            for name, value in after["conditions"].items()
        }
        if predictions_before != predictions_after:
            raise AssertionError("Adapter save and reload changed predictions")

        train_path = run_path / TRAIN_PATH
        removed_train_path = train_path.with_suffix(".removed")
        train_path.rename(removed_train_path)
        evaluation = evaluate_state_handoff_condition(
            run_path,
            "explicit_handoff",
            max_cases=2,
            model=reloaded,
            tokenizer=tokenizer,
        )
        return {
            "one_optimizer_step_completed": int(first["optimizer_step"]) == 1,
            "finite_losses": True,
            "adapter_roundtrip_preserved_predictions": True,
            "evaluation_without_training_dataset": int(evaluation["case_count"]) == 2,
            "resume_metric_steps": [row["optimizer_step"] for row in metrics],
            "duplicate_metrics_or_cases": False,
        }


def run_tiny_interface_smoke(_source_run_path: Path) -> dict[str, Any]:
    """Run one opaque-interface update and evaluate without training rows."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    from .state_interface_evaluation import evaluate_state_interface_condition

    repository_tmp = Path(__file__).resolve().parents[3] / ".tmp"
    repository_tmp.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="state_interface_smoke_", dir=repository_tmp
    ) as directory:
        run_path = Path(directory)
        _write_smoke_config(run_path)
        config_path = run_path / "config.yaml"
        config_path.write_text(
            config_path.read_text().replace(
                "state_handoff_training:\n",
                "state_handoff_training:\n"
                "  conditions: [canonical_opaque]\n"
                "  interfaces:\n"
                "    independent_module_contexts: false\n",
            )
        )
        prepare_state_handoff_datasets(run_path)
        tokenizer = _CharacterTokenizer()
        torch.manual_seed(11)
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=2048,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=1024,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=1,
            )
        )
        base_path = run_path / "tiny_base"
        model.config._name_or_path = str(base_path)
        model.config.save_pretrained(base_path)
        trained = train_state_handoff_condition(
            run_path,
            "canonical_opaque",
            max_optimizer_steps=1,
            model=model,
            tokenizer=tokenizer,
            enforce_phase1_gate=False,
        )
        metrics = read_training_metrics(run_path, "canonical_opaque")
        (run_path / TRAIN_PATH).rename(run_path / "training/data/train.removed")
        evaluation = evaluate_state_interface_condition(
            run_path,
            "canonical_opaque",
            max_cases=17,
            model=model.eval(),
            tokenizer=tokenizer,
        )
        return {
            "one_optimizer_step_completed": int(trained["optimizer_step"]) == 1,
            "finite_losses": all(
                math.isfinite(float(metrics[0][key]))
                for key in ("total_loss", "state_token_loss", "answer_loss")
            ),
            "evaluation_without_training_dataset": evaluation["case_count"] == 17,
            "recursive_h4_path_exercised": "4" in evaluation["by_horizon"],
            "matched_interface_compute": json.loads(
                (run_path / "training/compute_manifest.json").read_text()
            )["matched_forward_passes_and_tokens"],
        }
