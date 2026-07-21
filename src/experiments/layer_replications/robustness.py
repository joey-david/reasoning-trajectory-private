"""Lad et al. layer-drop and adjacent-layer-swap replication."""

from __future__ import annotations

from contextlib import contextmanager
import csv
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Iterator
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

import torch
import torch.nn.functional as F

from src.experiments.layer_replications.common import read_jsonl, replication_dir
from src.models.hf_loader import load_hf_tokenizer
from src.models.introspection import get_decoder_layers, get_input_device
from src.runtime.artifact_store import write_json
from src.runtime.config import load_config
from src.runtime.data import write_jsonl


KINDS = ("drop", "swap")


def results_path(run_path: Path) -> Path:
    """Return the append-only partial-metric artifact."""
    return replication_dir(run_path) / "lad_robustness/partials.jsonl"


def prepare_dataset(
    run_path: Path, *, token_limit: int | None = None
) -> dict[str, Any]:
    """Materialize fixed token blocks from the paper's Pile source."""
    from huggingface_hub import HfApi

    config = load_config(run_path)
    experiment = config["layer_robustness"]
    source = experiment["dataset"]
    model_cfg = config["model"]
    tokenizer = load_hf_tokenizer(model_cfg)
    target = int(token_limit or source["token_count"])
    sequence_length = int(source["sequence_length"])
    if target <= 0 or sequence_length <= 0:
        raise ValueError("token_count and sequence_length must be positive")

    revision = str(source["revision"])
    observed_revision = HfApi().dataset_info(source["path"]).sha
    if observed_revision != revision:
        raise RuntimeError(
            f"dataset head moved from pinned revision {revision} to {observed_revision}; "
            "refuse to materialize ambiguous Dataset Viewer rows"
        )
    rows: list[dict[str, Any]] = []
    scored = 0
    for sample in _dataset_viewer_rows(source):
        text = str(sample.get(source.get("text_field", "text"), ""))
        if not text:
            continue
        remaining = target - scored
        ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=min(sequence_length, remaining) + 1,
        )["input_ids"]
        if len(ids) < 2:
            continue
        current = len(ids) - 1
        rows.append(
            {
                "id": f"pile-{len(rows):06d}",
                "input_ids": ids,
                "scored_tokens": current,
            }
        )
        scored += current
        if scored >= target:
            break

    if scored != target:
        raise RuntimeError(f"Pile stream ended after {scored}/{target} scored tokens")
    write_jsonl(run_path / "dataset.jsonl", rows)
    manifest = {
        "source": source["path"],
        "source_revision": source.get("revision"),
        "tokenizer": model_cfg["name"],
        "tokenizer_revision": model_cfg.get("revision"),
        "sequence_length": sequence_length,
        "token_count": scored,
        "blocks": len(rows),
        "sampling": source.get("sampling", "source_order"),
        "sampling_seed": source.get("seed"),
    }
    write_json(
        replication_dir(run_path) / "lad_robustness/dataset_manifest.json", manifest
    )
    return manifest


def _dataset_viewer_rows(source: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Stream deterministic random pages without downloading Pile shards."""
    size_query = urlencode({"dataset": source["path"]})
    size_payload = _read_json_url(
        f"https://datasets-server.huggingface.co/size?{size_query}"
    )
    split = source.get("split", "train")
    split_sizes = {
        row["split"]: int(row["num_rows"])
        for row in size_payload["size"]["splits"]
        if row["config"] == (source.get("name") or "default")
    }
    total_rows = split_sizes[split]
    page_size = int(source.get("page_size", 100))
    rng = random.Random(int(source.get("seed", 0)))
    random_pages = source.get("sampling", "random_pages") == "random_pages"
    offset = rng.randrange(0, total_rows - page_size + 1) if random_pages else 0
    while True:
        query = urlencode(
            {
                "dataset": source["path"],
                "config": source.get("name") or "default",
                "split": source.get("split", "train"),
                "offset": offset,
                "length": page_size,
            }
        )
        payload = _read_json_url(
            f"https://datasets-server.huggingface.co/rows?{query}"
        )
        page = payload.get("rows", [])
        if not page:
            return
        for wrapped in page:
            yield dict(wrapped["row"])
        if random_pages:
            offset = rng.randrange(0, total_rows - page_size + 1)
        else:
            offset += len(page)
        time.sleep(float(source.get("request_interval_seconds", 2.1)))


def _read_json_url(url: str) -> dict[str, Any]:
    """Read one Dataset Viewer response with bounded rate-limit retries."""
    retryable = {429, 500, 502, 503, 504}
    for attempt in range(8):
        try:
            with urlopen(url, timeout=120) as response:
                return json.load(response)
        except HTTPError as exc:
            if exc.code not in retryable or attempt == 7:
                raise
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else min(30.0, 2.0**attempt)
            time.sleep(delay)
    raise AssertionError("unreachable")


def load_blocks(run_path: Path) -> list[dict[str, Any]]:
    """Load prepared token blocks."""
    rows = read_jsonl(run_path / "dataset.jsonl")
    if not rows:
        raise FileNotFoundError(
            f"{run_path / 'dataset.jsonl'} is absent or empty; run prepare-robustness"
        )
    return rows


def task_key(kind: str, layer: int, chunk: int) -> str:
    """Build a stable resumability key."""
    return f"{kind}:{layer}:{chunk}"


def expected_task_keys(run_path: Path) -> set[str]:
    """Enumerate the complete configured intervention matrix."""
    config = load_config(run_path)
    layer_count = int(config["model"]["layer_count"])
    blocks = load_blocks(run_path)
    per_chunk = int(config["layer_robustness"].get("blocks_per_task", 8))
    chunks = math.ceil(len(blocks) / per_chunk)
    keys: set[str] = set()
    for kind in KINDS:
        stop = layer_count - 1 if kind == "swap" else layer_count
        keys.update(
            task_key(kind, layer, chunk)
            for layer in range(stop)
            for chunk in range(chunks)
        )
    return keys


@contextmanager
def layer_intervention(
    model: torch.nn.Module, kind: str, layer_index: int
) -> Iterator[None]:
    """Temporarily drop one block or swap adjacent block execution order."""
    layers = get_decoder_layers(model)  # type: ignore[arg-type]
    if kind == "swap":
        if layer_index + 1 >= len(layers):
            raise IndexError("the final layer has no adjacent successor to swap")
        first, second = layers[layer_index], layers[layer_index + 1]
        layers[layer_index], layers[layer_index + 1] = second, first
        try:
            yield
        finally:
            layers[layer_index], layers[layer_index + 1] = first, second
        return

    if kind != "drop":
        raise ValueError(f"unknown intervention: {kind}")

    def preserve_residual(_module: Any, inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = inputs[0]
        if isinstance(output, tuple):
            return (hidden, *output[1:])
        return hidden

    handle = layers[layer_index].register_forward_hook(preserve_residual)
    try:
        yield
    finally:
        handle.remove()


def evaluate_chunk(
    model: torch.nn.Module,
    blocks: list[dict[str, Any]],
    *,
    kind: str,
    layer: int,
    chunk: int,
    blocks_per_task: int,
) -> dict[str, Any]:
    """Aggregate paper metrics over one resumable token-block chunk."""
    start = chunk * blocks_per_task
    selected = blocks[start : start + blocks_per_task]
    if not selected:
        raise IndexError(f"empty block chunk {chunk}")
    device = get_input_device(model)  # type: ignore[arg-type]
    totals = {
        "token_count": 0,
        "kl_sum": 0.0,
        "top1_match_sum": 0.0,
        "baseline_nll_sum": 0.0,
        "intervened_nll_sum": 0.0,
        "baseline_entropy_sum": 0.0,
        "intervened_entropy_sum": 0.0,
    }
    model.eval()
    with torch.inference_mode():
        for block in selected:
            ids = torch.tensor([block["input_ids"]], dtype=torch.long, device=device)
            baseline = model(input_ids=ids, use_cache=False).logits[:, :-1].float()
            with layer_intervention(model, kind, layer):
                intervened = (
                    model(input_ids=ids, use_cache=False).logits[:, :-1].float()
                )
            labels = ids[:, 1:]
            log_p = F.log_softmax(baseline, dim=-1)
            log_q = F.log_softmax(intervened, dim=-1)
            p = log_p.exp()
            q = log_q.exp()
            count = int(labels.numel())
            totals["token_count"] += count
            totals["kl_sum"] += float((p * (log_p - log_q)).sum().item())
            totals["top1_match_sum"] += float(
                (baseline.argmax(-1) == intervened.argmax(-1)).sum().item()
            )
            totals["baseline_nll_sum"] += float(
                F.cross_entropy(
                    baseline.reshape(-1, baseline.shape[-1]),
                    labels.reshape(-1),
                    reduction="sum",
                ).item()
            )
            totals["intervened_nll_sum"] += float(
                F.cross_entropy(
                    intervened.reshape(-1, intervened.shape[-1]),
                    labels.reshape(-1),
                    reduction="sum",
                ).item()
            )
            totals["baseline_entropy_sum"] += float((-(p * log_p).sum(-1)).sum().item())
            totals["intervened_entropy_sum"] += float(
                (-(q * log_q).sum(-1)).sum().item()
            )
    return {
        "key": task_key(kind, layer, chunk),
        "intervention": kind,
        "layer": layer,
        "chunk": chunk,
        **totals,
    }


def analyze(run_path: Path) -> dict[str, Any]:
    """Aggregate completed chunks into the paper's layer-wise curves."""
    rows = read_jsonl(results_path(run_path))
    expected = expected_task_keys(run_path)
    completed = {str(row["key"]) for row in rows}
    missing = sorted(expected - completed)
    if missing:
        raise RuntimeError(
            f"robustness matrix incomplete: {len(missing)}/{len(expected)} tasks remain"
        )

    grouped: dict[tuple[str, int], dict[str, float]] = {}
    for row in rows:
        key = (str(row["intervention"]), int(row["layer"]))
        bucket = grouped.setdefault(key, {})
        for field, value in row.items():
            if field.endswith("_sum") or field == "token_count":
                bucket[field] = bucket.get(field, 0.0) + float(value)

    curves: list[dict[str, Any]] = []
    for (kind, layer), sums in sorted(grouped.items()):
        count = int(sums["token_count"])
        curves.append(
            {
                "intervention": kind,
                "layer": layer,
                "token_count": count,
                "kl": sums["kl_sum"] / count,
                "top1_consistency": sums["top1_match_sum"] / count,
                "baseline_nll": sums["baseline_nll_sum"] / count,
                "intervened_nll": sums["intervened_nll_sum"] / count,
                "baseline_entropy_nats": sums["baseline_entropy_sum"] / count,
                "intervened_entropy_nats": sums["intervened_entropy_sum"] / count,
            }
        )

    out = replication_dir(run_path) / "lad_robustness"
    report = {
        "paper": "Lad et al., The Remarkable Robustness of LLMs",
        "complete": True,
        "tasks": len(expected),
        "curves": curves,
    }
    write_json(out / "report.json", report)
    with (out / "curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    _plot_curves(curves, out / "layer_interventions.png")
    return report


def _plot_curves(rows: list[dict[str, Any]], path: Path) -> None:
    """Render the two key intervention metrics without a plotting abstraction."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for kind in KINDS:
        selected = [row for row in rows if row["intervention"] == kind]
        axes[0].plot(
            [row["layer"] for row in selected],
            [row["kl"] for row in selected],
            marker="o",
            markersize=3,
            label=kind,
        )
        axes[1].plot(
            [row["layer"] for row in selected],
            [row["top1_consistency"] for row in selected],
            marker="o",
            markersize=3,
            label=kind,
        )
    axes[0].set(xlabel="layer", ylabel="KL(original || intervened)")
    axes[1].set(xlabel="layer", ylabel="top-1 consistency", ylim=(0, 1.02))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def validate(run_path: Path, *, require_dataset: bool = True) -> dict[str, Any]:
    """Validate the pinned model, paper protocol, and optional prepared corpus."""
    config = load_config(run_path)
    experiment = config["layer_robustness"]
    model = config["model"]
    checks = {
        "two_interventions": tuple(experiment.get("interventions", KINDS)) == KINDS,
        "positive_layer_count": int(model["layer_count"]) > 1,
        "paper_token_budget": int(experiment["dataset"]["token_count"]) == 1_000_000,
        "pinned_model_revision": bool(model.get("revision")),
        "pinned_dataset_revision": bool(experiment["dataset"].get("revision")),
    }
    if require_dataset:
        blocks = load_blocks(run_path)
        manifest_path = (
            replication_dir(run_path) / "lad_robustness/dataset_manifest.json"
        )
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {}
        )
        checks["dataset_token_budget"] = sum(
            int(row["scored_tokens"]) for row in blocks
        ) == int(experiment["dataset"]["token_count"])
        checks["random_sequence_sampling"] = (
            manifest.get("sampling") == "random_pages"
            and manifest.get("sampling_seed") == experiment["dataset"].get("seed")
            and all(
                len(row["input_ids"]) == int(row["scored_tokens"]) + 1
                and int(row["scored_tokens"])
                <= int(experiment["dataset"]["sequence_length"])
                for row in blocks
            )
        )
    if not all(checks.values()):
        raise ValueError(f"invalid robustness replication: {checks}")
    return {"checks": checks, "valid": True}
