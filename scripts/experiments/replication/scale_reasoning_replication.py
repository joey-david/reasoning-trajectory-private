#!/usr/bin/env python3
"""Prepare and drive larger-Qwen reasoning-trajectory replications."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.analysis.dataset_screening import summarize_run
from src.runtime.config import load_config


ROOT = Path(__file__).resolve().parents[3]
PYTHON = ".venv/bin/python"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    aliases: tuple[str, ...]
    run_dir: str
    hf_name: str
    revision: str | None
    hidden_layers: int
    think_end_token_id: int
    component_layer: int
    component_layers: tuple[int, ...]
    default_devices: str
    required_gpus: int = 1
    attn_implementation: str | None = "flash_attention_2"
    screening_rollouts: int = 10
    primary_rollouts: int = 10
    max_new_tokens: int = 8192


SPECS = (
    ModelSpec(
        key="qwen3-8b",
        aliases=("qwen3-8b", "Qwen3-8B", "Qwen/Qwen3-8B"),
        run_dir="Qwen3-8B",
        hf_name="Qwen/Qwen3-8B",
        revision="b968826d9c46dd6066d109eabc6255188de91218",
        hidden_layers=36,
        think_end_token_id=151668,
        component_layer=24,
        component_layers=(12, 24, 35),
        default_devices="0",
        max_new_tokens=16384,
    ),
    ModelSpec(
        key="qwen3.6-27b",
        aliases=("qwen3.6-27b", "Qwen3.6-27B", "Qwen/Qwen3.6-27B"),
        run_dir="Qwen3.6-27B",
        hf_name="Qwen/Qwen3.6-27B",
        revision=None,
        hidden_layers=64,
        think_end_token_id=248069,
        component_layer=42,
        component_layers=(21, 42, 63),
        default_devices="0+1",
        required_gpus=2,
        attn_implementation=None,
        screening_rollouts=8,
        primary_rollouts=8,
        max_new_tokens=12288,
    ),
)


SCREEN_TEMPLATES = {
    "aime_2024_frontier_screen": Path(
        "runs/Qwen3-14B/screening/aime_2024_frontier_screen/config.yaml"
    ),
    "aime_2025_frontier_screen": Path(
        "runs/Qwen3-14B/screening/aime_2025_frontier_screen/config.yaml"
    ),
    "olympiadbench_math_frontier_screen": Path(
        "runs/Qwen3-14B/screening/olympiadbench_math_frontier_screen/config.yaml"
    ),
    "gpqa_diamond_hard_screen": Path(
        "runs/Qwen3-14B/screening/gpqa_diamond_hard_screen/config.yaml"
    ),
    "math_algebra_hard_screen": Path(
        "runs/Qwen3-14B/screening/math_algebra_hard_screen/config.yaml"
    ),
    "polymath_high_frontier_screen": Path(
        "runs/Qwen3-8B/screening/polymath_high_numeric_screen/config.yaml"
    ),
}


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def model_spec(name: str) -> ModelSpec:
    lowered = name.lower()
    for spec in SPECS:
        if lowered in {alias.lower() for alias in spec.aliases}:
            return spec
    choices = ", ".join(spec.key for spec in SPECS)
    raise SystemExit(f"unknown model {name!r}; choose one of: {choices}")


def model_config(spec: ModelSpec) -> dict[str, Any]:
    config: dict[str, Any] = {
        "name": spec.hf_name,
        "backend": "hf",
        "device_map": "auto",
        "dtype": "bfloat16",
        "trust_remote_code": True,
    }
    if spec.revision:
        config["revision"] = spec.revision
    if spec.attn_implementation:
        config["attn_implementation"] = spec.attn_implementation
    if spec.required_gpus > 1:
        config["required_gpus"] = spec.required_gpus
    return config


def run_root(spec: ModelSpec) -> Path:
    return Path("runs") / spec.run_dir


def screening_runs(spec: ModelSpec) -> list[Path]:
    base = run_root(spec) / "screening"
    return [base / name for name in SCREEN_TEMPLATES]


def primary_run(spec: ModelSpec) -> Path:
    return (
        run_root(spec)
        / "screening"
        / "frontier_identification"
        / "frontier_reasoning_mixed"
    )


def h1_runs(spec: ModelSpec) -> dict[str, Path]:
    base = run_root(spec) / "pilots"
    return {
        "freeform": base / "h1_freeform_replay",
        "numbered": base / "h1_numbered_steps_pilot",
        "sentence": base / "h1_sentence_separated_pilot",
        "paragraph": base / "h1_paragraph_separated_pilot",
    }


def h2_component_replay(spec: ModelSpec) -> Path:
    return run_root(spec) / "replay" / "h2_component_replay"


def h4_replay(spec: ModelSpec) -> Path:
    return run_root(spec) / "replay" / "h4_structural_replay"


def gold_run(spec: ModelSpec) -> Path:
    return run_root(spec) / "replay" / "thought_units_gold_answers"


def h3_replay(spec: ModelSpec) -> Path:
    return run_root(spec) / "failed" / "h3_process_isomer_replay"


def h3_patch_runs(spec: ModelSpec) -> dict[str, Path]:
    base = run_root(spec) / "failed"
    return {
        "attention_output": base / "h3_process_isomer_patching",
        "mlp_output": base / "h3_process_isomer_patching_mlp",
    }


def solution_object_runs(spec: ModelSpec) -> dict[str, Path]:
    base = run_root(spec) / "interventions"
    return {
        "small": base / "solution_object_extraction_small",
        "medium": base / "solution_object_extraction_medium",
    }


def label_run(spec: ModelSpec) -> Path:
    return (
        Path("runs")
        / "Qwen3.5-122B-A10B-FP8"
        / "labeling"
        / f"solution_object_silver_{spec.key.replace('.', '_')}"
    )


def experiment_dir(spec: ModelSpec) -> Path:
    return Path("experiments") / spec.key.replace(".", "_")


def configure_common_analysis(config: dict[str, Any], spec: ModelSpec) -> None:
    analysis = config.setdefault("analysis", {})
    analysis["think_end_token_id"] = spec.think_end_token_id
    analysis.setdefault("hard_question_limit", 80)
    analysis.setdefault(
        "produced_answer_regex",
        "(?i)(?:\\\\boxed\\{|(?:Final\\s+)?Answer\\s*:\\s*)"
        "(-?\\d[\\d,]*(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)",
    )
    analysis.setdefault(
        "gold_answer_regex",
        "####\\s*(-?\\d[\\d,]*(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)",
    )


def prepare_screening_configs(spec: ModelSpec) -> None:
    for name, template in SCREEN_TEMPLATES.items():
        config = read_yaml(ROOT / template)
        config["model"] = model_config(spec)
        generation = config.setdefault("generation", {})
        generation["num_samples_per_item"] = max(
            int(generation.get("num_samples_per_item", 1)), spec.screening_rollouts
        )
        generation["max_new_tokens"] = max(
            int(generation.get("max_new_tokens", 1024)), spec.max_new_tokens
        )
        generation["base_seed"] = stable_seed(spec, name)
        configure_common_analysis(config, spec)
        write_yaml(ROOT / run_root(spec) / "screening" / name / "config.yaml", config)


def prepare_primary_config(spec: ModelSpec) -> None:
    config = read_yaml(
        ROOT
        / "runs/SmolLM3-3B/screening/frontier_identification/"
        "gsm_symb_pure_mixed_latents_10k/config.yaml"
    )
    run_path = primary_run(spec)
    config["model"] = model_config(spec)
    config["dataset"] = {
        "source": "jsonl",
        "path": (run_path / "dataset.jsonl").as_posix(),
        "adapter": "plain_question",
        "sample_limit": 80,
        "sample_offset": 0,
        "shuffle_seed": None,
        "require_gold_answer": True,
    }
    config["generation"]["num_samples_per_item"] = spec.primary_rollouts
    config["generation"]["max_new_tokens"] = spec.max_new_tokens
    config["generation"]["base_seed"] = stable_seed(spec, "frontier_reasoning_mixed")
    config["generation"].pop("forced_prefix", None)
    config["generation"].pop("cap_fallback", None)
    config["capture"] = {
        "enabled": True,
        "layers": [-1],
        "activation_storage_dtype": "int8_scaled",
        "diagnostics": False,
    }
    configure_common_analysis(config, spec)
    write_yaml(ROOT / run_path / "config.yaml", config)


def prepare_h1_configs(spec: ModelSpec) -> None:
    runs = h1_runs(spec)
    source = primary_run(spec)
    numbered_dataset = runs["numbered"] / "dataset.jsonl"
    templates = {
        "freeform": Path("runs/SmolLM3-3B/pilots/h1_freeform_replay/config.yaml"),
        "numbered": Path("runs/SmolLM3-3B/pilots/h1_numbered_steps_pilot/config.yaml"),
        "sentence": Path("runs/SmolLM3-3B/pilots/h1_sentence_separated_pilot/config.yaml"),
        "paragraph": Path("runs/SmolLM3-3B/pilots/h1_paragraph_separated_pilot/config.yaml"),
    }
    for condition, template in templates.items():
        config = read_yaml(ROOT / template)
        config["model"] = model_config(spec)
        if condition == "freeform":
            config["replay"]["source_run"] = source.as_posix()
            config["replay"]["sample_ids_path"] = numbered_dataset.as_posix()
            config["replay"]["per_sample"] = min(5, spec.primary_rollouts)
            config["replay"]["max_trajectories"] = 80
        else:
            config["dataset"]["path"] = (source / "dataset.jsonl").as_posix()
            config["dataset"]["sample_limit"] = 16
            config["generation"]["num_samples_per_item"] = 5
            config["generation"]["max_new_tokens"] = min(spec.max_new_tokens, 8192)
            config["generation"]["base_seed"] = stable_seed(spec, f"h1_{condition}")
            config["generation"].pop("forced_prefix", None)
            config["generation"].pop("cap_fallback", None)
        config["capture"]["layers"] = [-1]
        configure_common_analysis(config, spec)
        write_yaml(ROOT / runs[condition] / "config.yaml", config)


def prepare_replay_configs(spec: ModelSpec) -> None:
    common_capture = {
        "enabled": True,
        "components": ["mlp_output", "attention_output"],
        "activation_storage_dtype": "int8_scaled",
        "diagnostics": False,
    }
    h2_config = {
        "model": model_config(spec),
        "replay": {
            "source_run": primary_run(spec).as_posix(),
            "per_sample": min(3, spec.primary_rollouts),
            "max_trajectories": 240,
        },
        "capture": {**common_capture, "layers": list(spec.component_layers)},
    }
    write_yaml(ROOT / h2_component_replay(spec) / "config.yaml", h2_config)

    h4_config = {
        "model": model_config(spec),
        "replay": {
            "source_run": primary_run(spec).as_posix(),
            "per_sample": min(5, spec.primary_rollouts),
            "max_trajectories": 400,
        },
        "capture": {
            "enabled": True,
            "layers": [-1],
            "components": [],
            "activation_storage_dtype": "int8_scaled",
            "diagnostics": False,
        },
    }
    write_yaml(ROOT / h4_replay(spec) / "config.yaml", h4_config)

    gold_config = {
        "model": model_config(spec),
        "dataset": {
            "source": "jsonl",
            "path": (gold_run(spec) / "dataset.jsonl").as_posix(),
            "adapter": "plain_question",
            "sample_limit": 80,
            "sample_offset": 0,
            "shuffle_seed": None,
            "require_gold_answer": True,
        },
        "generation": {"max_new_tokens": spec.max_new_tokens},
        "gold_answer_capture": {
            "layers": [-1],
            "max_tokens": spec.max_new_tokens,
            "activation_storage_dtype": "int8_scaled",
        },
    }
    write_yaml(ROOT / gold_run(spec) / "config.yaml", gold_config)


def prepare_h3_configs(spec: ModelSpec) -> None:
    out = experiment_dir(spec)
    pairs = out / "h3_process_isomer_pairs.jsonl"
    audit = out / "h3_process_isomer_pair_audit.json"
    replay = {
        "model": model_config(spec),
        "replay": {
            "source_run": primary_run(spec).as_posix(),
            "pair_manifest_path": pairs.as_posix(),
            "per_sample": spec.primary_rollouts,
        },
        "capture": {
            "enabled": True,
            "layers": [spec.component_layer],
            "components": ["mlp_output", "attention_output"],
            "activation_storage_dtype": "int8_scaled",
            "diagnostics": False,
        },
    }
    write_yaml(ROOT / h3_replay(spec) / "config.yaml", replay)

    patch_templates = {
        "attention_output": "h3_process_isomer_patching",
        "mlp_output": "h3_process_isomer_patching_mlp18",
    }
    for component, template_name in patch_templates.items():
        config = read_yaml(
            ROOT / "runs/SmolLM3-3B/failed" / template_name / "config.yaml"
        )
        config["model"] = model_config(spec)
        patching = config["patching"]
        patching["activation_run"] = h3_replay(spec).as_posix()
        patching["pairs"] = pairs.as_posix()
        patching["pair_audit"] = audit.as_posix()
        patching["component"] = component
        patching["layer"] = spec.component_layer
        patching["projection_path"] = (
            out / "h3_projections" / f"{component}_layer{spec.component_layer}_projection.pt"
        ).as_posix()
        patching["projection_report"] = (
            out / "h3_projections" / f"{component}_layer{spec.component_layer}_report.json"
        ).as_posix()
        patching["continuations_per_condition"] = 5
        patching["max_new_tokens"] = min(spec.max_new_tokens, 1536)
        patching["base_seed"] = stable_seed(spec, f"h3_{component}")
        configure_common_analysis(config, spec)
        write_yaml(ROOT / h3_patch_runs(spec)[component] / "config.yaml", config)


def prepare_solution_object_configs(spec: ModelSpec) -> None:
    for scale, template in {
        "small": Path(
            "runs/SmolLM3-3B/interventions/solution_object_extraction_small/config.yaml"
        ),
        "medium": Path(
            "runs/SmolLM3-3B/interventions/solution_object_extraction_medium/config.yaml"
        ),
    }.items():
        config = read_yaml(ROOT / template)
        config["model"] = model_config(spec)
        experiment = config["solution_object_extraction"]
        experiment["capture"]["layers"] = list(spec.component_layers)
        experiment["mixed_trajectories"]["source_run"] = primary_run(spec).as_posix()
        experiment["mixed_trajectories"]["layer"] = -1
        experiment["mixed_trajectories"]["per_sample"] = spec.primary_rollouts
        write_yaml(ROOT / solution_object_runs(spec)[scale] / "config.yaml", config)


def prepare_label_config(spec: ModelSpec) -> None:
    target = label_run(spec)
    config = read_yaml(
        ROOT / "runs/Qwen3.5-122B-A10B-FP8/labeling/solution_object_silver/config.yaml"
    )
    config["solution_object_labeling"]["input"] = (
        target / "token_windows.jsonl"
    ).as_posix()
    config["solution_object_labeling"]["output"] = (
        target / "labels" / "silver_labels.jsonl"
    ).as_posix()
    write_yaml(ROOT / target / "config.yaml", config)


def prepare_configs(spec: ModelSpec) -> None:
    prepare_screening_configs(spec)
    prepare_primary_config(spec)
    prepare_h1_configs(spec)
    prepare_replay_configs(spec)
    prepare_h3_configs(spec)
    prepare_solution_object_configs(spec)
    prepare_label_config(spec)


def stable_seed(spec: ModelSpec, label: str) -> int:
    value = 0
    for char in f"{spec.key}:{label}":
        value = (value * 131 + ord(char)) % 9_000_000
    return 1_000_000 + value


def orchestrate_cmd(
    run_path: Path,
    job: str,
    args: argparse.Namespace,
    spec: ModelSpec,
) -> list[str]:
    return [
        PYTHON,
        "scripts/orchestrate.py",
        "--job",
        job,
        "--nodes",
        args.nodes,
        "--devices",
        args.devices or spec.default_devices,
        "--run",
        run_path.as_posix(),
    ]


def direct_model_cmd(
    spec: ModelSpec,
    args: argparse.Namespace,
    command: list[str],
) -> list[str]:
    """Prefix direct model-loading commands with the selected visible GPUs."""
    if spec.required_gpus <= 1:
        return command
    first_group = (args.devices or spec.default_devices).split(",")[0]
    visible = first_group.replace("+", ",")
    return [
        "env",
        f"CUDA_VISIBLE_DEVICES={visible}",
        f"ORCHESTRATOR_GPU_COUNT={spec.required_gpus}",
        *command,
    ]


def command_plan(spec: ModelSpec, args: argparse.Namespace) -> list[list[str]]:
    primary = primary_run(spec)
    h2_dir = primary / "analysis/experiments/h2_localized_updates"
    h4_h2_dir = h4_replay(spec) / "analysis/experiments/h2_localized_updates"
    out = experiment_dir(spec)
    pairs = out / "h3_process_isomer_pairs.jsonl"
    audit = out / "h3_process_isomer_pair_audit.json"
    projection_dir = out / "h3_projections"
    labels = label_run(spec)
    commands: list[list[str]] = []

    if enabled(args, "screening"):
        for run_path in screening_runs(spec):
            commands.extend(
                [
                    [PYTHON, "scripts/data/prepare_dataset.py", run_path.as_posix()],
                    orchestrate_cmd(run_path, "generation", args, spec),
                    [PYTHON, "scripts/analysis/analyze.py", run_path.as_posix()],
                ]
            )
        commands.append(
            [
                PYTHON,
                "scripts/analysis/summarize_screening.py",
                *[path.as_posix() for path in screening_runs(spec)],
            ]
        )
        commands.append(
            [
                PYTHON,
                "scripts/experiments/replication/scale_reasoning_replication.py",
                spec.key,
                "select-frontier",
            ]
        )

    if enabled(args, "primary"):
        commands.extend(
            [
                orchestrate_cmd(primary, "generation", args, spec),
                [PYTHON, "scripts/analysis/analyze.py", primary.as_posix()],
            ]
        )

    if enabled(args, "experiments"):
        h1 = h1_runs(spec)
        commands.extend(
            [
                direct_model_cmd(
                    spec,
                    args,
                    [PYTHON, "scripts/experiments/trajectory_dynamics/replay_capture.py", h1["freeform"].as_posix()],
                ),
                *[
                    orchestrate_cmd(h1[name], "generation", args, spec)
                    for name in ("numbered", "sentence", "paragraph")
                ],
                [
                    PYTHON,
                    "scripts/experiments/boundaries/boundary_comparison.py",
                    *(run.as_posix() for run in h1.values()),
                ],
                [PYTHON, "scripts/experiments/trajectory_dynamics/localized_updates.py", primary.as_posix()],
                [
                    PYTHON,
                    "scripts/experiments/trajectory_dynamics/correctness_prediction.py",
                    primary.as_posix(),
                ],
                orchestrate_cmd(gold_run(spec), "gold_answer_capture", args, spec),
                [
                    PYTHON,
                    "scripts/experiments/token_segmentation/token_segmentation.py",
                    primary.as_posix(),
                    "--gold-run",
                    gold_run(spec).as_posix(),
                    "--updates",
                    (h2_dir / "updates.jsonl").as_posix(),
                ],
                direct_model_cmd(
                    spec,
                    args,
                    [PYTHON, "scripts/experiments/trajectory_dynamics/replay_capture.py", h4_replay(spec).as_posix()],
                ),
                [
                    PYTHON,
                    "scripts/experiments/trajectory_dynamics/localized_updates.py",
                    h4_replay(spec).as_posix(),
                    "--per-sample",
                    "5",
                ],
                [
                    PYTHON,
                    "scripts/experiments/trajectory_dynamics/structural_contrast.py",
                    h4_h2_dir.as_posix(),
                ],
            ]
        )

    if enabled(args, "labeling"):
        commands.extend(
            [
                [
                    PYTHON,
                    "scripts/experiments/solution_object_extraction/prepare_solution_object_labels.py",
                    primary.as_posix(),
                    (labels / "token_windows.jsonl").as_posix(),
                    "--updates",
                    (h2_dir / "updates.jsonl").as_posix(),
                ],
                orchestrate_cmd(labels, "solution_object_labeling", args, spec),
                [
                    PYTHON,
                    "scripts/experiments/token_segmentation/semantic_token_segmentation.py",
                    primary.as_posix(),
                    "--labels-run",
                    labels.as_posix(),
                    "--gold-run",
                    gold_run(spec).as_posix(),
                    "--updates",
                    (h2_dir / "updates.jsonl").as_posix(),
                ],
            ]
        )

    if enabled(args, "h3"):
        commands.extend(
            [
                [
                    PYTHON,
                    "scripts/experiments/process_isomers/mine_process_isomers.py",
                    h2_dir.as_posix(),
                    pairs.as_posix(),
                    "--activation-run",
                    primary.as_posix(),
                    "--generation-run",
                    primary.as_posix(),
                    "--audit-path",
                    audit.as_posix(),
                    "--per-sample",
                    str(spec.primary_rollouts),
                ],
                direct_model_cmd(
                    spec,
                    args,
                    [PYTHON, "scripts/experiments/trajectory_dynamics/replay_capture.py", h3_replay(spec).as_posix()],
                ),
                [
                    PYTHON,
                    "scripts/experiments/process_isomers/component_localization.py",
                    h2_component_replay(spec).as_posix(),
                    h2_dir.as_posix(),
                ],
                [
                    PYTHON,
                    "scripts/experiments/process_isomers/component_projection.py",
                    h2_component_replay(spec).as_posix(),
                    h2_dir.as_posix(),
                    projection_dir.as_posix(),
                    "--layer",
                    str(spec.component_layer),
                ],
                *[
                    [
                        PYTHON,
                        "scripts/experiments/process_isomers/causal_patching.py",
                        run_path.as_posix(),
                        "--validate-only",
                    ]
                    for run_path in h3_patch_runs(spec).values()
                ],
                *[
                    orchestrate_cmd(run_path, "causal_patching", args, spec)
                    for run_path in h3_patch_runs(spec).values()
                ],
                *[
                    [
                        PYTHON,
                        "scripts/experiments/process_isomers/analyze_causal_patching.py",
                        run_path.as_posix(),
                    ]
                    for run_path in h3_patch_runs(spec).values()
                ],
            ]
        )

    if enabled(args, "solution-objects"):
        medium = solution_object_runs(spec)["medium"]
        commands.extend(
            [
                direct_model_cmd(
                    spec,
                    args,
                    [
                        PYTHON,
                        "scripts/experiments/solution_object_extraction/solution_object_extraction.py",
                        "prepare",
                        medium.as_posix(),
                    ],
                ),
                direct_model_cmd(
                    spec,
                    args,
                    [
                        PYTHON,
                        "scripts/experiments/solution_object_extraction/solution_object_extraction.py",
                        "run",
                        medium.as_posix(),
                    ],
                ),
                direct_model_cmd(
                    spec,
                    args,
                    [
                        PYTHON,
                        "scripts/experiments/solution_object_extraction/solution_object_extraction.py",
                        "improve",
                        medium.as_posix(),
                    ],
                ),
                direct_model_cmd(
                    spec,
                    args,
                    [
                        PYTHON,
                        "scripts/experiments/solution_object_extraction/solution_object_extraction.py",
                        "validate-improvement",
                        medium.as_posix(),
                    ],
                ),
            ]
        )

    return commands


def enabled(args: argparse.Namespace, stage: str) -> bool:
    return not args.stages or stage in set(args.stages)


def print_plan(commands: list[list[str]]) -> None:
    for index, command in enumerate(commands, start=1):
        print(f"{index:03d}  {shell_join(command)}")


def shell_join(command: list[str]) -> str:
    import shlex

    return " ".join(shlex.quote(part) for part in command)


def run_commands(commands: list[list[str]], dry_run: bool) -> None:
    for command in commands:
        print(f"+ {shell_join(command)}", flush=True)
        if not dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


def select_frontier(spec: ModelSpec, max_pass_rate: float) -> Path:
    candidates = []
    for run_path in screening_runs(spec):
        generation_index = ROOT / run_path / "generation" / "generations.jsonl"
        if not generation_index.exists():
            continue
        summary = summarize_run(run_path)
        if summary["status"] != "completed":
            continue
        candidates.append(summary)
    if not candidates:
        raise SystemExit(
            f"{spec.key}: no completed screening runs available; run screening first"
        )
    candidates.sort(
        key=lambda row: (
            int(row["frontier_instances"]),
            int(row["mixed_instances"]),
            float(row["accuracy"] or 0.0),
        ),
        reverse=True,
    )
    best = Path(candidates[0]["run_path"])
    out = primary_run(spec) / "dataset.jsonl"
    subprocess.run(
        [
            PYTHON,
            "scripts/analysis/select_mixed_samples.py",
            best.as_posix(),
            "--out",
            out.as_posix(),
            "--max-pass-rate",
            str(max_pass_rate),
        ],
        cwd=ROOT,
        check=True,
    )
    copy_selected_dataset(spec, out)
    write_frontier_choice(spec, candidates)
    return out


def copy_selected_dataset(spec: ModelSpec, dataset_path: Path) -> None:
    targets = [
        h1_runs(spec)["numbered"] / "dataset.jsonl",
        h1_runs(spec)["sentence"] / "dataset.jsonl",
        h1_runs(spec)["paragraph"] / "dataset.jsonl",
        gold_run(spec) / "dataset.jsonl",
    ]
    for target in targets:
        (ROOT / target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / dataset_path, ROOT / target)


def write_frontier_choice(spec: ModelSpec, summaries: list[dict[str, Any]]) -> None:
    path = ROOT / experiment_dir(spec) / "frontier_screening_candidates.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_path",
        "classification",
        "accuracy",
        "frontier_instances",
        "mixed_instances",
        "capped_rollout_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in summaries:
            writer.writerow({field: row.get(field, "") for field in fields})


def validate_prepared_configs(spec: ModelSpec) -> None:
    paths = [
        *screening_runs(spec),
        primary_run(spec),
        *h1_runs(spec).values(),
        h2_component_replay(spec),
        h4_replay(spec),
        gold_run(spec),
        h3_replay(spec),
        *h3_patch_runs(spec).values(),
        *solution_object_runs(spec).values(),
        label_run(spec),
    ]
    missing = [path for path in paths if not (ROOT / path / "config.yaml").exists()]
    if missing:
        raise SystemExit("missing configs:\n" + "\n".join(path.as_posix() for path in missing))
    for path in paths:
        load_config(ROOT / path)
    help_commands = [
        [PYTHON, "scripts/experiments/boundaries/boundary_comparison.py", "--help"],
        [PYTHON, "scripts/experiments/token_segmentation/token_segmentation.py", "--help"],
        [PYTHON, "scripts/experiments/token_segmentation/semantic_token_segmentation.py", "--help"],
        [PYTHON, "scripts/experiments/process_isomers/causal_patching.py", "--help"],
        [PYTHON, "scripts/experiments/solution_object_extraction/solution_object_extraction.py", "--help"],
    ]
    for command in help_commands:
        subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    print(f"{spec.key}: validated {len(paths)} configs and {len(help_commands)} CLIs")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare and run larger-Qwen reasoning-trajectory replications."
    )
    parser.add_argument("model", help="qwen3-8b or qwen3.6-27b")
    parser.add_argument(
        "command",
        nargs="?",
        choices=("prepare", "plan", "validate", "select-frontier", "run"),
        default="plan",
    )
    parser.add_argument("--nodes", default="upnquick")
    parser.add_argument("--devices", help="Orchestrator device expression.")
    parser.add_argument(
        "--stages",
        nargs="*",
        choices=(
            "screening",
            "primary",
            "experiments",
            "labeling",
            "h3",
            "solution-objects",
        ),
        help="Limit the command plan to selected stages.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-pass-rate", type=float, default=0.85)
    args = parser.parse_args()

    spec = model_spec(args.model)
    prepare_configs(spec)

    if args.command == "prepare":
        print(f"{spec.key}: prepared configs under {run_root(spec)}")
        return 0
    if args.command == "validate":
        validate_prepared_configs(spec)
        return 0
    if args.command == "select-frontier":
        print(select_frontier(spec, args.max_pass_rate))
        return 0

    commands = command_plan(spec, args)
    if args.command == "plan":
        print_plan(commands)
    else:
        run_commands(commands, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
