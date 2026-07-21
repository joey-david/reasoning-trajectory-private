"""Zhang et al. single-layer GRPO replication and contribution analysis."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
from typing import Any

import numpy as np
import torch

from src.experiments.layer_replications.common import replication_dir
from src.models.introspection import get_decoder_layers
from src.runtime.artifact_store import write_json
from src.runtime.config import load_config


def setting_name(layer: int | None, *, full: bool = False) -> str:
    """Return the stable name for a base/full/single-layer setting."""
    if full:
        return "full"
    return "base" if layer is None else f"layer-{layer:02d}"


def checkpoint_dir(run_path: Path, setting: str) -> Path:
    """Return the remote-owned heavy checkpoint directory."""
    return replication_dir(run_path) / "zhang_single_layer_rl/checkpoints" / setting


def trainable_state_path(run_path: Path, setting: str) -> Path:
    """Return the compact decoder-only state retained after training."""
    return checkpoint_dir(run_path, setting) / "trainable.safetensors"


def evaluation_path(run_path: Path, setting: str) -> Path:
    """Return one setting's compact benchmark report."""
    return (
        replication_dir(run_path)
        / "zhang_single_layer_rl/evaluations"
        / f"{setting}.json"
    )


def configure_trainable_layers(
    model: torch.nn.Module, *, layer: int | None = None, full: bool = False
) -> dict[str, Any]:
    """Freeze embeddings/head and expose only the requested decoder parameter subspace."""
    if full == (layer is not None):
        raise ValueError("choose exactly one of full decoder training or one layer")
    model.requires_grad_(False)
    layers = get_decoder_layers(model)  # type: ignore[arg-type]
    selected = range(len(layers)) if full else (int(layer),)
    for index in selected:
        if index < 0 or index >= len(layers):
            raise IndexError(f"layer {index} outside [0, {len(layers)})")
        layers[index].requires_grad_(True)
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "setting": setting_name(layer, full=full),
        "selected_layers": list(selected),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": trainable / total,
        "embeddings_frozen": not any(
            parameter.requires_grad
            for parameter in model.get_input_embeddings().parameters()
        ),
        "lm_head_frozen": not any(
            parameter.requires_grad
            for parameter in model.get_output_embeddings().parameters()
        ),
    }


def _training_dataset(config: dict[str, Any]) -> Any:
    """Load and normalize the pinned NuminaMath-CoT training split."""
    from datasets import load_dataset

    dataset_cfg = config["dataset"]
    dataset = load_dataset(
        dataset_cfg["path"],
        dataset_cfg.get("name"),
        split=dataset_cfg.get("split", "train"),
        revision=dataset_cfg.get("revision"),
    )
    limit = int(dataset_cfg.get("sample_limit", len(dataset)))
    dataset = dataset.select(range(min(limit, len(dataset))))

    def normalize(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "prompt": [{"role": "user", "content": str(row["problem"])}],
            "solution": str(row["solution"]),
        }

    return dataset.map(normalize, remove_columns=dataset.column_names)


def build_grpo_config(
    model_cfg: dict[str, Any],
    training: dict[str, Any],
    output: Path,
    *,
    world_size: int,
) -> Any:
    """Translate the paper's rollout, mini-batch, and micro-batch contract to TRL."""
    from trl import GRPOConfig

    micro_batch = int(training["micro_batch_size"])
    mini_batch = int(training["ppo_mini_batch_size"])
    denominator = micro_batch * world_size
    if mini_batch % denominator:
        raise ValueError(
            f"ppo_mini_batch_size {mini_batch} is not divisible by micro batch "
            f"{micro_batch} x world size {world_size}"
        )
    rollout_batch = int(training["train_batch_size"]) * int(training["group_size"])
    return GRPOConfig(
        output_dir=str(output),
        model_init_kwargs={
            "revision": model_cfg.get("revision"),
            "dtype": "bfloat16",
            "attn_implementation": model_cfg.get("attn_implementation", "sdpa"),
        },
        trust_remote_code=True,
        learning_rate=float(training["learning_rate"]),
        num_train_epochs=float(training["epochs"]),
        per_device_train_batch_size=micro_batch,
        gradient_accumulation_steps=mini_batch // denominator,
        generation_batch_size=rollout_batch,
        num_generations=int(training["group_size"]),
        max_completion_length=int(training["max_response_length"]),
        beta=float(training["kl_coefficient"]),
        epsilon=float(training["clip_range"]),
        loss_type="grpo",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        logging_steps=int(training.get("logging_steps", 1)),
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        use_transformers_continuous_batching=bool(
            training.get("use_transformers_continuous_batching", True)
        ),
        seed=int(training["seed"]),
    )


def train(
    run_path: Path,
    *,
    layer: int | None = None,
    full: bool = False,
    resume: bool = True,
) -> dict[str, Any]:
    """Run one paper-matched GRPO setting through the maintained TRL trainer."""
    try:
        from safetensors.torch import save_file
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOTrainer
        from trl.rewards import accuracy_reward
    except ImportError as exc:
        raise RuntimeError(
            "single-layer RL needs the pinned optional dependencies; "
            "run scripts/experiments/install_layer_rl_deps.sh"
        ) from exc

    config = load_config(run_path)
    model_cfg = config["model"]
    experiment = config["single_layer_rl"]
    training = experiment["training"]
    setting = setting_name(layer, full=full)
    output = checkpoint_dir(run_path, setting)
    final_marker = output / "replication_complete.json"
    if final_marker.exists():
        state_path = trainable_state_path(run_path, setting)
        if not state_path.exists():
            raise FileNotFoundError(
                f"completed marker exists without trained decoder state: {state_path}"
            )
        return json.loads(final_marker.read_text(encoding="utf-8"))

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"], revision=model_cfg.get("revision"), trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        revision=model_cfg.get("revision"),
        dtype=torch.bfloat16,
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
        trust_remote_code=True,
    )
    trainable = configure_trainable_layers(model, layer=layer, full=full)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args = build_grpo_config(
        model_cfg,
        training,
        output,
        world_size=world_size,
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=accuracy_reward,
        args=args,
        train_dataset=_training_dataset(experiment),
    )
    checkpoint = _latest_checkpoint(output) if resume else None
    trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    trainable_state = {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    save_file(trainable_state, trainable_state_path(run_path, setting))
    metadata = {
        **trainable,
        "model": model_cfg["name"],
        "model_revision": model_cfg.get("revision"),
        "dataset": experiment["dataset"]["path"],
        "dataset_revision": experiment["dataset"].get("revision"),
        "trl_version": _package_version("trl"),
        "transformers_version": _package_version("transformers"),
        "complete": True,
    }
    for path in output.glob("checkpoint-*"):
        if path.is_dir():
            shutil.rmtree(path)
    write_json(final_marker, metadata)
    return metadata


def _latest_checkpoint(output: Path) -> Path | None:
    """Find the highest numbered Trainer checkpoint for restart safety."""
    candidates = []
    for path in output.glob("checkpoint-*"):
        try:
            candidates.append((int(path.name.rsplit("-", 1)[1]), path))
        except ValueError:
            continue
    return max(candidates, default=(0, None))[1]


def _package_version(name: str) -> str:
    """Read an installed distribution version for provenance."""
    from importlib.metadata import version

    return version(name)


def _evaluation_specs(experiment: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the four in-domain benchmark specifications from the paper."""
    return [dict(spec) for spec in experiment["evaluation"]["benchmarks"]]


def _load_evaluation_rows(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize one pinned Hugging Face benchmark to problem/solution rows."""
    from datasets import load_dataset

    dataset = load_dataset(
        spec["path"],
        spec.get("name"),
        split=spec.get("split", "test"),
        revision=spec.get("revision"),
    )
    answer_pattern = (
        re.compile(str(spec["answer_regex"]), re.DOTALL)
        if spec.get("answer_regex")
        else None
    )
    rows = []
    for row in dataset:
        problem = str(row[spec["problem_field"]])
        solution = row[spec["answer_field"]]
        solutions = solution if isinstance(solution, list) else [solution]
        normalized_solutions = []
        for value in solutions:
            text = str(value)
            if answer_pattern is not None:
                match = answer_pattern.search(text)
                if match is None:
                    raise ValueError(
                        f"{spec['key']} answer does not match answer_regex: {text!r}"
                    )
                text = match.group("answer")
            normalized_solutions.append(text)
        rows.append(
            {"problem": problem, "solutions": normalized_solutions}
        )
    limit = int(spec.get("sample_limit", len(rows)))
    return rows[:limit]


def _chat_prompt(tokenizer: Any, problem: str) -> str:
    """Apply the Qwen reasoning template, with a stable base-model fallback."""
    messages = [
        {
            "role": "user",
            "content": (
                "Solve the problem step by step. Put the final answer in \\boxed{}.\n\n"
                + problem
            ),
        }
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    return messages[0]["content"] + "\n"


def evaluate(
    run_path: Path,
    *,
    layer: int | None = None,
    full: bool = False,
    base: bool = False,
) -> dict[str, Any]:
    """Evaluate base/full/single-layer models on the four paper math benchmarks."""
    try:
        from safetensors.torch import load_file
        from tqdm.auto import tqdm
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl.rewards import accuracy_reward
    except ImportError as exc:
        raise RuntimeError(
            "single-layer RL evaluation needs the pinned optional dependencies"
        ) from exc

    if sum((base, full, layer is not None)) != 1:
        raise ValueError("choose exactly one of base, full, or layer")
    config = load_config(run_path)
    experiment = config["single_layer_rl"]
    evaluation = experiment["evaluation"]
    setting = "base" if base else setting_name(layer, full=full)
    destination = evaluation_path(run_path, setting)
    if destination.exists():
        return json.loads(destination.read_text(encoding="utf-8"))
    model_path = config["model"]["name"]
    revision = config["model"].get("revision")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, revision=revision, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        revision=revision,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation=config["model"].get("attn_implementation", "sdpa"),
        trust_remote_code=True,
    ).eval()
    if not base:
        state_path = trainable_state_path(run_path, setting)
        if not state_path.exists():
            raise FileNotFoundError(f"trained decoder state is missing: {state_path}")
        state = load_file(state_path)
        parameters = dict(model.named_parameters())
        unexpected = sorted(set(state) - set(parameters))
        if unexpected:
            raise KeyError(
                f"trained state contains unknown parameters: {unexpected[:3]}"
            )
        with torch.no_grad():
            for name, value in state.items():
                parameters[name].copy_(
                    value.to(
                        device=parameters[name].device, dtype=parameters[name].dtype
                    )
                )
    device = model.get_input_embeddings().weight.device
    benchmark_reports = []
    for spec in _evaluation_specs(experiment):
        rows = _load_evaluation_rows(spec)
        sample_count = int(spec.get("samples_per_problem", 1))
        rewards: list[float] = []
        batch_size = int(evaluation["batch_size"])
        for start in tqdm(
            range(0, len(rows), batch_size),
            desc=f"{setting} {spec['key']}",
            unit="batch",
            leave=False,
        ):
            batch = rows[start : start + batch_size]
            prompts = [_chat_prompt(tokenizer, row["problem"]) for row in batch]
            expanded_prompts = [
                prompt for prompt in prompts for _ in range(sample_count)
            ]
            encoded = tokenizer(expanded_prompts, return_tensors="pt", padding=True).to(
                device
            )
            torch.manual_seed(int(evaluation["seed"]) + start)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(evaluation["seed"]) + start)
            generation_kwargs = {
                "max_new_tokens": int(evaluation["max_new_tokens"]),
                "do_sample": sample_count > 1,
            }
            if sample_count > 1:
                generation_kwargs.update(
                    {
                        "temperature": float(spec.get("temperature", 0.6)),
                        "top_p": float(spec.get("top_p", 0.95)),
                    }
                )
            with torch.inference_mode():
                generated = model.generate(**encoded, **generation_kwargs)
            completions = []
            prompt_width = int(encoded["input_ids"].shape[1])
            for sequence in generated:
                text = tokenizer.decode(
                    sequence[prompt_width:], skip_special_tokens=True
                )
                completions.append([{"role": "assistant", "content": text}])
            solutions = [row["solutions"] for row in batch for _ in range(sample_count)]
            for completion, accepted in zip(completions, solutions, strict=True):
                reward = max(
                    float(accuracy_reward([completion], [solution])[0] or 0.0)
                    for solution in accepted
                )
                rewards.append(reward)
        score = float(np.mean(rewards)) if rewards else float("nan")
        benchmark_reports.append(
            {
                "name": spec["key"],
                "problems": len(rows),
                "samples_per_problem": sample_count,
                "score": score,
            }
        )
    report = {
        "setting": setting,
        "model_path": model_path,
        "benchmarks": benchmark_reports,
        "math_average": float(np.mean([row["score"] for row in benchmark_reports])),
        "complete": True,
    }
    write_json(destination, report)
    return report


def validate(run_path: Path) -> dict[str, Any]:
    """Validate paper model, data, training, and evaluation contracts."""
    config = load_config(run_path)
    experiment = config["single_layer_rl"]
    training = experiment["training"]
    benchmarks = experiment["evaluation"]["benchmarks"]
    checks = {
        "paper_model": config["model"]["name"] == "Qwen/Qwen3-1.7B-Base",
        "full_layer_scan": int(config["model"]["layer_count"]) == 28,
        "numina_50k": int(experiment["dataset"]["sample_limit"]) == 50_000,
        "paper_learning_rate": float(training["learning_rate"]) == 5e-6,
        "paper_train_batch": int(training["train_batch_size"]) == 512,
        "paper_ppo_mini_batch": int(training["ppo_mini_batch_size"]) == 128,
        "paper_micro_batch": int(training["micro_batch_size"]) == 8,
        "paper_group_size": int(training["group_size"]) == 4,
        "paper_response_length": int(training["max_response_length"]) == 3072,
        "paper_kl": float(training["kl_coefficient"]) == 0.001,
        "paper_clip": float(training["clip_range"]) == 0.2,
        "paper_epochs": int(training["epochs"]) == 4,
        "four_math_benchmarks": [row["key"] for row in benchmarks]
        == ["math500", "gsm8k", "olympiadbench", "amc"],
        "gsm8k_final_answer_extraction": "(?P<answer>"
        in str(benchmarks[1].get("answer_regex", "")),
        "amc_average_32": int(benchmarks[-1]["samples_per_problem"]) == 32,
        "pinned_model_revision": bool(config["model"].get("revision")),
        "pinned_dataset_revision": bool(experiment["dataset"].get("revision")),
    }
    if not all(checks.values()):
        raise ValueError(f"invalid single-layer RL replication: {checks}")
    return {"checks": checks, "valid": True}
