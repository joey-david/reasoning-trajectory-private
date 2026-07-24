"""Shared supervised LoRA trainer for outcome and explicit-handoff conditions."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import time
from typing import Any, Callable

from src.runtime.artifact_store import append_jsonl, write_json
from src.runtime.config import load_config

from .state_handoff_data import (
    COMPUTE_MANIFEST_PATH,
    DATA_MANIFEST_PATH,
    TRAINING_CONDITIONS,
    TRAIN_PATH,
    VALIDATION_PATH,
    build_training_pairs,
    matched_compute_manifest,
    read_programs,
    configured_training_conditions,
)
from .state_handoff_training_runtime import (
    _base_and_adapter,
    _collate,
    _condition_loss,
    _evaluate_training_pairs,
    _load_resume_state,
    _pair_batches,
    _per_sequence_loss,
    _save_checkpoint,
)
from .state_interface_contract import is_interface_condition


def condition_training_dir(run_path: Path, condition: str) -> Path:
    """Return the artifact owner for one training condition."""
    if condition not in TRAINING_CONDITIONS and not is_interface_condition(condition):
        raise ValueError(f"Unknown state-handoff training condition: {condition!r}")
    return run_path / "training" / condition


def validation_checkpoint_score(
    validation: dict[str, dict[str, float]] | None, condition: str
) -> tuple[str, float | None]:
    """Score checkpoints on the target needed from the trained producer."""
    metric = "answer" if condition == "outcome_only" else "state"
    values = [
        horizon[metric]
        for horizon in (validation or {}).values()
        if metric in horizon
    ]
    return metric, (sum(values) / len(values) if values else None)


def require_phase1_training_gate(run_path: Path) -> None:
    """Block adapter training until Phase 1 passes and the 7B screen completes."""
    config = load_config(run_path).get("state_handoff_training", {})
    source = Path(str(config["phase1_source_run"]))
    summary_path = source / "depth_relief/explicit_handoff/summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"Missing Phase 1 summary: {summary_path}")
    status = json.loads(summary_path.read_text())["phase1_gate"]["status"]
    if status != "passed":
        raise RuntimeError(f"Phase 1 gate is {status!r}; LoRA training is blocked")
    screen_source = Path(str(config.get("frozen_screen_source_run", run_path)))
    factorization_path = screen_source / "depth_relief/factorization_summary.json"
    screen_path = screen_source / "depth_relief/explicit_handoff/summary.json"
    if not factorization_path.exists() or not screen_path.exists():
        raise RuntimeError("Frozen 7B factorization and handoff screens must run first")
    screen = json.loads(screen_path.read_text())
    if int(screen.get("phase_counts", {}).get("inference", 0)) != int(
        screen.get("case_count", -1)
    ):
        raise RuntimeError("Frozen 7B explicit-handoff screen is incomplete")


def read_training_metrics(run_path: Path, condition: str) -> list[dict[str, Any]]:
    """Read append-only optimizer metrics and reject resumed duplicates."""
    path = condition_training_dir(run_path, condition) / "metrics.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    steps = [int(row["optimizer_step"]) for row in rows]
    if len(steps) != len(set(steps)):
        raise ValueError(f"Duplicate optimizer steps in {path}")
    if steps != sorted(steps):
        raise ValueError(f"Optimizer steps are not ordered in {path}")
    return rows


def _flush_checkpoint_metrics(
    run_path: Path, condition: str, pending: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Append checkpoint-owned metrics once and recover a partial flush."""
    rows = read_training_metrics(run_path, condition)
    by_step = {int(row["optimizer_step"]): row for row in rows}
    last_step = int(rows[-1]["optimizer_step"]) if rows else 0
    path = condition_training_dir(run_path, condition) / "metrics.jsonl"
    for metric in pending:
        step = int(metric["optimizer_step"])
        existing = by_step.get(step)
        if existing is not None:
            if existing != metric:
                raise ValueError(f"Checkpoint metric changed at optimizer step {step}")
            continue
        if step != last_step + 1:
            raise ValueError(f"Checkpoint metrics have a gap before optimizer step {step}")
        append_jsonl(path, metric)
        rows.append(metric)
        by_step[step] = metric
        last_step = step
    return rows


def train_state_handoff_condition(
    run_path: Path,
    condition: str,
    *,
    max_optimizer_steps: int | None = None,
    model: Any | None = None,
    tokenizer: Any | None = None,
    enforce_phase1_gate: bool = True,
    on_progress: Callable[[str], None] | None = None,
    on_step: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Train or resume one matched-compute LoRA condition."""
    if condition not in TRAINING_CONDITIONS and not is_interface_condition(condition):
        raise ValueError(f"Unknown state-handoff training condition: {condition!r}")
    if enforce_phase1_gate:
        require_phase1_training_gate(run_path)
    import torch
    from transformers import get_linear_schedule_with_warmup

    config = load_config(run_path)
    experiment = config.get("state_handoff_training", {})
    training = experiment.get("training", {})
    seed = int(training.get("seed", 721_401))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    output = condition_training_dir(run_path, condition)
    manifest_path = output / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest.get("status") == "complete" and max_optimizer_steps is None:
        return manifest
    checkpoint = Path(manifest["last_checkpoint"]) if manifest.get("last_checkpoint") else None
    model, tokenizer = _base_and_adapter(
        run_path=run_path,
        condition=condition,
        checkpoint=checkpoint,
        model=model,
        tokenizer=tokenizer,
    )
    if bool(training.get("gradient_checkpointing", True)):
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    device = model.get_input_embeddings().weight.device
    max_length = int(training.get("max_sequence_length", 256))
    fixed_sequence_padding = bool(training.get("fixed_sequence_padding", False))
    microbatch = int(training.get("microbatch", 8))
    accumulation = int(training.get("gradient_accumulation", 8))
    if microbatch % 2:
        raise ValueError("Microbatch must preserve complete two-sequence program pairs")
    if int(training.get("effective_batch", microbatch * accumulation)) != (
        microbatch * accumulation
    ):
        raise ValueError("Configured effective batch does not match microbatch accumulation")
    if str(training.get("optimizer", "adamw")).lower() != "adamw":
        raise ValueError("State-handoff training supports only AdamW")
    epochs = int(training.get("epochs", 2))
    if not 1 <= epochs <= 8:
        raise ValueError("State-handoff training is capped at eight epochs")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_cases = read_programs(run_path / TRAIN_PATH)
    validation_cases = read_programs(run_path / VALIDATION_PATH)
    prompt = experiment.get("prompt", {})
    if condition in TRAINING_CONDITIONS:
        pairs = build_training_pairs(
            tokenizer=tokenizer,
            cases=train_cases,
            prompt_config=prompt,
            condition=condition,
            max_length=max_length,
            fixed_sequence_padding=fixed_sequence_padding,
        )
        validation_pairs = build_training_pairs(
            tokenizer=tokenizer,
            cases=validation_cases,
            prompt_config=prompt,
            condition=condition,
            max_length=max_length,
            fixed_sequence_padding=fixed_sequence_padding,
        )
    else:
        from .state_interface_data import build_interface_training_pairs

        interface = experiment.get("interfaces", {})
        pairs = build_interface_training_pairs(
            tokenizer=tokenizer,
            cases=train_cases,
            prompt_config=prompt,
            condition=condition,
            interface_config=interface,
            max_length=max_length,
        )
        validation_pairs = build_interface_training_pairs(
            tokenizer=tokenizer,
            cases=validation_cases,
            prompt_config=prompt,
            condition=condition,
            interface_config=interface,
            max_length=max_length,
        )
    compute_path = run_path / COMPUTE_MANIFEST_PATH
    if not compute_path.exists():
        if condition in TRAINING_CONDITIONS:
            compute = matched_compute_manifest(
                tokenizer=tokenizer,
                cases=train_cases,
                prompt_config=prompt,
                max_length=max_length,
                fixed_sequence_padding=fixed_sequence_padding,
            )
        else:
            from .state_interface_data import matched_interface_compute_manifest

            conditions = configured_training_conditions(run_path)
            if not all(is_interface_condition(value) for value in conditions):
                raise ValueError("Interface runs cannot mix terminal and code conditions")
            compute = matched_interface_compute_manifest(
                tokenizer=tokenizer,
                cases=train_cases,
                prompt_config=prompt,
                conditions=conditions,
                interface_config=experiment.get("interfaces", {}),
                max_length=max_length,
            )
        if not compute["matched_forward_passes_and_tokens"]:
            raise RuntimeError("Training conditions do not have matched compute")
        write_json(compute_path, compute)
    pair_batch_size = microbatch // 2
    microbatches_per_epoch = math.ceil(len(pairs) / pair_batch_size)
    total_microbatches = microbatches_per_epoch * epochs
    planned_total_steps = math.ceil(total_microbatches / accumulation)
    stop_after_steps = (
        min(planned_total_steps, int(max_optimizer_steps))
        if max_optimizer_steps is not None
        else planned_total_steps
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training.get("learning_rate", 1e-4)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    warmup_steps = math.ceil(
        float(training.get("warmup_ratio", 0.03)) * planned_total_steps
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, planned_total_steps
    )
    state = {"optimizer_step": 0, "microbatch_index": 0, "best_validation_accuracy": -1.0}
    if checkpoint is not None:
        state = _load_resume_state(
            model=model, optimizer=optimizer, scheduler=scheduler, checkpoint=checkpoint
        )
    pending_metrics = list(state.pop("pending_metrics", []))
    metrics = _flush_checkpoint_metrics(run_path, condition, pending_metrics)
    if metrics and int(metrics[-1]["optimizer_step"]) != int(state["optimizer_step"]):
        raise ValueError("Checkpoint and metrics disagree on the last optimizer step")
    skip_microbatches = int(state["microbatch_index"])
    batches = _pair_batches(
        pairs, pair_batch_size=pair_batch_size, epochs=epochs, seed=seed
    )
    optimizer.zero_grad(set_to_none=True)
    running: defaultdict[str, float] = defaultdict(float)
    running_counts: defaultdict[str, int] = defaultdict(int)
    interval_started = time.monotonic()
    eval_interval = int(training.get("evaluation_interval", 250))
    if eval_interval < 1:
        raise ValueError("Evaluation interval must be positive")
    max_grad_norm = float(training.get("max_gradient_norm", 1.0))
    checkpoint_metrics: list[dict[str, Any]] = []

    def report_step(loss: float | None = None) -> None:
        if on_step is None:
            return
        microbatch_index = int(state["microbatch_index"])
        epoch = min(
            epochs,
            microbatch_index // microbatches_per_epoch + 1,
        )
        description = (
            f"state handoff training {condition} "
            f"epoch {epoch}/{epochs}"
        )
        if loss is not None:
            description += f" loss={loss:.4f}"
        on_step(
            int(state["optimizer_step"]),
            stop_after_steps,
            description,
        )

    report_step()
    for microbatch_index, pair_batch in enumerate(batches):
        if microbatch_index < skip_microbatches:
            continue
        if int(state["optimizer_step"]) >= stop_after_steps:
            break
        sequences = [sequence for pair in pair_batch for sequence in pair]
        batch = _collate(
            sequences=sequences,
            tokenizer=tokenizer,
            max_length=max_length,
            device=device,
        )
        output_model = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
            return_dict=True,
        )
        losses, correct = _per_sequence_loss(output_model.logits, batch["labels"])
        total_loss, state_loss, answer_loss = _condition_loss(
            losses, batch["mappings"], condition
        )
        (total_loss / accumulation).backward()
        running["total_loss"] += float(total_loss.detach())
        running["answer_loss"] += float(answer_loss.detach())
        running_counts["microbatches"] += 1
        if state_loss is not None:
            running["state_loss"] += float(state_loss.detach())
            running_counts["state_loss"] += 1
        for mapping, value in zip(batch["mappings"], correct.tolist()):
            running[f"{mapping}_correct"] += int(value)
            running_counts[f"{mapping}_correct"] += 1
        running_counts["examples"] += len(sequences)
        state["microbatch_index"] = microbatch_index + 1
        boundary = (
            running_counts["microbatches"] == accumulation
            or state["microbatch_index"] == total_microbatches
        )
        if not boundary:
            continue
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        state["optimizer_step"] = int(state["optimizer_step"]) + 1
        elapsed = max(time.monotonic() - interval_started, 1e-9)
        step = int(state["optimizer_step"])
        should_evaluate = step % eval_interval == 0 or step == stop_after_steps
        if should_evaluate and on_progress is not None:
            on_progress(
                f"state handoff training {condition} validating at step {step}"
            )
        validation = (
            _evaluate_training_pairs(
                model=model,
                tokenizer=tokenizer,
                pairs=validation_pairs,
                max_length=max_length,
                microbatch=microbatch,
            )
            if should_evaluate
            else None
        )
        selection_metric, validation_score = validation_checkpoint_score(
            validation, condition
        )
        metric = {
            "optimizer_step": step,
            "total_loss": running["total_loss"] / running_counts["microbatches"],
            "state_token_loss": (
                running["state_loss"] / running_counts["state_loss"]
                if running_counts["state_loss"]
                else None
            ),
            "answer_loss": running["answer_loss"] / running_counts["microbatches"],
            "state_accuracy": (
                running["state_correct"] / running_counts["state_correct"]
                if running_counts["state_correct"]
                else None
            ),
            "answer_accuracy": running["answer_correct"] / running_counts["answer_correct"],
            "validation_accuracy_by_horizon": validation,
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "gradient_norm": float(gradient_norm),
            "examples_per_second": running_counts["examples"] / elapsed,
            "tokens_per_second": running_counts["examples"] * max_length / elapsed,
            "wall_time_seconds": elapsed,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
            ),
        }
        if not all(
            math.isfinite(float(value))
            for key, value in metric.items()
            if key in {"total_loss", "answer_loss", "gradient_norm"}
            or (key == "state_token_loss" and value is not None)
        ):
            raise FloatingPointError("State-handoff training produced a non-finite metric")
        checkpoint_metrics.append(metric)
        if on_progress is not None and on_step is None:
            on_progress(
                f"state handoff training {condition} step {step}/{stop_after_steps} "
                f"loss={metric['total_loss']:.4f}"
            )
        report_step(metric["total_loss"])
        running.clear()
        running_counts.clear()
        interval_started = time.monotonic()
        if should_evaluate:
            checkpoint_dir = output / "checkpoints" / f"step_{step:06d}"
            if validation_score is not None and validation_score > float(
                state["best_validation_accuracy"]
            ):
                state["best_validation_accuracy"] = validation_score
                state["best_validation_metric"] = selection_metric
                state["best_checkpoint"] = str(checkpoint_dir)
            state["pending_metrics"] = checkpoint_metrics
            _save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                directory=checkpoint_dir,
                state=state,
            )
            manifest = {
                "schema_version": 1,
                "condition": condition,
                "status": "running",
                "last_checkpoint": str(checkpoint_dir),
                "best_checkpoint": state.get("best_checkpoint"),
                "optimizer_step": step,
                "dataset_manifest": str(run_path / DATA_MANIFEST_PATH),
                "base_model": config["model"]["name"],
                "base_model_revision": config["model"].get("revision"),
                "initial_adapter": (
                    experiment.get("interfaces", {})
                    .get("initial_adapters", {})
                    .get(condition)
                ),
                "training_sequences_per_epoch": 2 * len(pairs),
                "planned_training_forward_passes": 2 * len(pairs) * epochs,
                "planned_fixed_padding_compute_tokens": (
                    2 * len(pairs) * epochs * max_length
                ),
            }
            write_json(manifest_path, manifest)
            _flush_checkpoint_metrics(run_path, condition, checkpoint_metrics)
            checkpoint_metrics = []
            state.pop("pending_metrics", None)
            if on_progress is not None:
                on_progress(
                    f"state handoff training {condition} saved step {step}"
                )
    final_adapter = output / "adapter" / "final"
    model.save_pretrained(final_adapter, safe_serialization=True)
    manifest = {
        **manifest,
        "status": (
            "complete"
            if int(state["optimizer_step"]) >= planned_total_steps
            else "partial"
        ),
        "optimizer_step": int(state["optimizer_step"]),
        "final_adapter": str(final_adapter),
    }
    write_json(manifest_path, manifest)
    return manifest


def state_handoff_training_status(run_path: Path) -> dict[str, Any]:
    """Report data and per-condition checkpoint state without loading a model."""
    conditions = {}
    for condition in configured_training_conditions(run_path):
        path = condition_training_dir(run_path, condition) / "checkpoint_manifest.json"
        conditions[condition] = json.loads(path.read_text()) if path.exists() else {
            "status": "not_started"
        }
    return {
        "run_path": str(run_path),
        "data_prepared": (run_path / DATA_MANIFEST_PATH).exists(),
        "compute_manifest_exists": (run_path / COMPUTE_MANIFEST_PATH).exists(),
        "conditions": conditions,
    }
