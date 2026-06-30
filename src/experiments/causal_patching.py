"""Run component-level causal patches between symbolic process isomers."""

from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import random
from typing import Any

import torch

from src.analysis.answers import answers_match, extract_answer
from src.analysis.common import read_generation_rows
from src.experiments.replay_capture import load_source_sample
from src.models.generation_pipeline import set_seed
from src.models.hf_loader import load_hf_model_and_tokenizer
from src.models.introspection import (
    get_decoder_layers,
    get_input_device,
    resolve_layer_indices,
)
from src.runtime.artifact_store import append_jsonl, load_component_states_npz
from src.runtime.config import load_config
from src.runtime.data import load_samples


def run_causal_patching(run_path: Path) -> None:
    """Execute H3 conditions over a prepared process-isomer pair manifest."""
    config = load_config(run_path)
    patch_cfg = config["patching"]
    activation_run = Path(patch_cfg["activation_run"])
    pairs = load_samples(patch_cfg["pairs"])
    pairs = pairs[: int(patch_cfg.get("max_pairs", len(pairs)))]
    component, layer = resolve_patch_target(patch_cfg)
    model, tokenizer = load_hf_model_and_tokenizer(config["model"])
    rows = {
        (str(row["sample_id"]), int(row["seed"])): row
        for row in read_generation_rows(activation_run)
    }
    output_path = run_path / "patching" / "continuations.jsonl"
    completed = load_completed_patches(output_path)
    vector_cache: dict[tuple[str, int, int], torch.Tensor] = {}

    for pair_index, pair in enumerate(pairs):
        for condition in patch_cfg["conditions"]:
            for continuation in range(
                int(patch_cfg.get("continuations_per_condition", 5))
            ):
                key = (int(pair["pair_id"]), str(condition), continuation)
                if key in completed:
                    continue
                seed = (
                    int(patch_cfg.get("base_seed", 0)) + pair_index * 100 + continuation
                )
                record = generate_patched_continuation(
                    model=model,
                    tokenizer=tokenizer,
                    activation_run=activation_run,
                    rows=rows,
                    pairs=pairs,
                    pair=pair,
                    condition=str(condition),
                    continuation=continuation,
                    seed=seed,
                    component=component,
                    layer=layer,
                    patch_cfg=patch_cfg,
                    analysis_cfg=config.get("analysis", {}),
                    vector_cache=vector_cache,
                )
                append_jsonl(output_path, record)
                completed.add(key)


def generate_patched_continuation(
    *,
    model: Any,
    tokenizer: Any,
    activation_run: Path,
    rows: dict[tuple[str, int], dict[str, Any]],
    pairs: list[dict[str, Any]],
    pair: dict[str, Any],
    condition: str,
    continuation: int,
    seed: int,
    component: str,
    layer: int,
    patch_cfg: dict[str, Any],
    analysis_cfg: dict[str, Any],
    vector_cache: dict[tuple[str, int, int], torch.Tensor],
) -> dict[str, Any]:
    target = pair["target"]
    target_key = (str(target["sample_id"]), int(target["seed"]))
    target_row = rows[target_key]
    sample = load_source_sample(activation_run, target_key[0])
    generated_ids = [int(token) for token in target_row["generated_token_ids"]]
    token_end = min(int(target["token_end"]), len(generated_ids) - 1)
    prefix_ids = [*sample["input_ids"], *generated_ids[: token_end + 1]]

    patch_vector = None
    donor_description = None
    if condition != "baseline":
        donor_point, donor_state_index = select_control_donor(
            condition=condition,
            pair=pair,
            pairs=pairs,
            rows=rows,
            target_row=target_row,
        )
        donor_key = (
            str(donor_point["sample_id"]),
            int(donor_point["seed"]),
        )
        cache_key = (donor_key[0], donor_key[1], donor_state_index)
        if cache_key not in vector_cache:
            donor_row = rows[donor_key]
            component_states, layers = load_component_states_npz(
                activation_run / donor_row["hidden_states_file"],
                component,
            )
            layer_col = layers.index(layer)
            state_index = min(max(donor_state_index, 0), len(component_states) - 1)
            vector_cache[cache_key] = torch.from_numpy(
                component_states[state_index, layer_col].astype("float32")
            )
        patch_vector = vector_cache[cache_key]
        donor_description = {
            "sample_id": donor_key[0],
            "seed": donor_key[1],
            "state_index": donor_state_index,
        }

    set_seed(seed)
    input_device = get_input_device(model)
    input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=input_device)
    attention_mask = torch.ones_like(input_ids)
    patch_context = (
        FirstForwardComponentPatch(
            model=model,
            layer=layer,
            component=component,
            vector=patch_vector,
        )
        if patch_vector is not None
        else nullcontext()
    )
    with patch_context:
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(patch_cfg.get("max_new_tokens", 768)),
            do_sample=float(patch_cfg.get("temperature", 0.0)) > 0,
            temperature=float(patch_cfg.get("temperature", 0.6)),
            top_p=float(patch_cfg.get("top_p", 0.95)),
            top_k=int(patch_cfg.get("top_k", 20)),
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    continuation_ids = output[0, len(prefix_ids) :].detach().cpu().tolist()
    text = tokenizer.decode(
        continuation_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    produced_answer = extract_answer(text, analysis_cfg.get("produced_answer_regex"))
    gold_answer = extract_answer(
        str(sample.get("gold_answer", "")),
        analysis_cfg.get("gold_answer_regex"),
    )
    return {
        "pair_id": pair["pair_id"],
        "condition": condition,
        "continuation": continuation,
        "seed": seed,
        "component": component,
        "layer": layer,
        "target": target,
        "donor": donor_description,
        "generated_token_ids": continuation_ids,
        "produced_text": text,
        "produced_answer": produced_answer,
        "gold_answer": gold_answer,
        "is_correct": answers_match(produced_answer, gold_answer),
        "has_valid_answer": produced_answer is not None,
    }


def select_control_donor(
    *,
    condition: str,
    pair: dict[str, Any],
    pairs: list[dict[str, Any]],
    rows: dict[tuple[str, int], dict[str, Any]],
    target_row: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    if condition == "equivalent":
        donor = pair["donor"]
        return donor, int(donor["state_index"])

    alternatives = [
        candidate
        for candidate in pairs
        if candidate["pair_id"] != pair["pair_id"]
        and candidate["graph_signature"] != pair["graph_signature"]
    ]
    if not alternatives:
        raise ValueError("H3 controls require at least two distinct graph states")
    candidate = random.Random(int(pair["pair_id"])).choice(alternatives)
    donor = candidate["donor"]
    if condition == "mismatched":
        return donor, int(donor["state_index"])
    if condition == "position_random":
        target_fraction = int(pair["target"]["state_index"]) / max(
            len(target_row["generated_token_ids"]), 1
        )
        donor_row = rows[(str(donor["sample_id"]), int(donor["seed"]))]
        position = round(
            target_fraction * max(len(donor_row["generated_token_ids"]) - 1, 0)
        )
        return donor, int(position)
    raise ValueError(f"Unknown patching condition: {condition!r}")


def resolve_patch_target(patch_cfg: dict[str, Any]) -> tuple[str, int]:
    component = patch_cfg.get("component")
    layer = patch_cfg.get("layer")
    if component != "auto" and layer != "auto":
        return str(component), int(layer)
    report = json.loads(Path(patch_cfg["component_report"]).read_text())
    target = report.get("recommended_patch_target")
    if not target:
        raise ValueError("Component localization has no recommended patch target")
    return str(target["component"]), int(target["layer"])


def load_completed_patches(path: Path) -> set[tuple[int, str, int]]:
    if not path.exists():
        return set()
    return {
        (int(row["pair_id"]), str(row["condition"]), int(row["continuation"]))
        for row in load_samples(path.resolve())
    }


class FirstForwardComponentPatch:
    """Replace one component's final prefill-token output exactly once."""

    def __init__(
        self,
        *,
        model: Any,
        layer: int,
        component: str,
        vector: torch.Tensor,
    ) -> None:
        decoder_layers = get_decoder_layers(model)
        resolved = resolve_layer_indices([layer], len(decoder_layers))[0]
        attribute = "mlp" if component == "mlp_output" else "self_attn"
        self.module = getattr(decoder_layers[resolved], attribute)
        self.vector = vector
        self.handle = None
        self.applied = False

    def __enter__(self) -> FirstForwardComponentPatch:
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def _hook(self, _module, _inputs, output):
        if self.applied:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        patched = hidden.clone()
        patched[:, -1, :] = self.vector.to(
            device=patched.device,
            dtype=patched.dtype,
        )
        self.applied = True
        if isinstance(output, tuple):
            return (patched, *output[1:])
        return patched

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.handle is not None:
            self.handle.remove()
