"""Multitask nonlinear object encoder with contrastive and adversarial variants."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from tqdm.auto import tqdm

from .decoders import decode_representation
from .projections import project, random_projection
from .retrieval import evaluate_by_split
from .sweeps import (
    EVAL_SPLITS,
    load_feature_views,
    nuisance_probe_accuracy,
    read_json,
    with_template_validation,
)
from .storage import (
    load_experiment_config,
    output_dir,
    read_jsonl,
    write_json,
)


EDIT_ORDER = {"BIND": "OPERATE", "OPERATE": "VERIFY", "VERIFY": "EXTRACT", "EXTRACT": "END"}


@dataclass(slots=True)
class LabelBundle:
    graph: np.ndarray
    edit: np.ndarray
    operation: np.ndarray
    value: np.ndarray
    role: np.ndarray
    next_edit: np.ndarray
    lexical: np.ndarray
    template: np.ndarray
    classes: dict[str, list[str]]


class GradientReverse(torch.autograd.Function):
    """Identity forward with a sign-reversed backward gradient."""

    @staticmethod
    def forward(ctx: Any, values: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return values.view_as(values)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.scale * gradient, None


class ObjectEncoder(torch.nn.Module):
    """Shared MLP encoder and factorized object/nuisance heads."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        class_counts: dict[str, int],
        base_dim: int = 0,
    ) -> None:
        super().__init__()
        if base_dim > latent_dim:
            raise ValueError("The residual base cannot exceed the latent dimension")
        self.base_dim = base_dim
        self.latent_dim = latent_dim
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, latent_dim),
        )
        if base_dim:
            torch.nn.init.zeros_(self.encoder[-1].weight)
            torch.nn.init.zeros_(self.encoder[-1].bias)
        self.heads = torch.nn.ModuleDict(
            {
                name: torch.nn.Linear(latent_dim, count)
                for name, count in class_counts.items()
            }
        )

    def forward(
        self, values: torch.Tensor, *, adversarial_scale: float
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z = self.encoder(values)
        if self.base_dim:
            base = torch.nn.functional.pad(
                values[:, : self.base_dim],
                (0, self.latent_dim - self.base_dim),
            )
            z = base + 0.1 * z
        normalized = torch.nn.functional.normalize(z, dim=1)
        outputs = {}
        for name, head in self.heads.items():
            head_input = (
                GradientReverse.apply(normalized, adversarial_scale)
                if name in {"lexical", "template"}
                else normalized
            )
            outputs[name] = head(head_input)
        return normalized, outputs


def run_nonlinear_sweep(run_path: Path, *, local: bool) -> dict[str, Any]:
    """Train and compare MLP, contrastive, and adversarial object encoders."""
    loaded = load_experiment_config(run_path)
    cfg = loaded["experiment"]["improvement"]["nonlinear"]
    improvement = loaded["experiment"]["improvement"]
    records = read_jsonl(run_path / "dataset.jsonl")
    selection_records = with_template_validation(records)
    views, captured_layers, _ = load_feature_views(
        output_dir(run_path) / "captured_features.npz"
    )
    retrieval = read_json(
        output_dir(run_path) / "improvement" / "retrieval_sweep.json"
    )
    selected_linear = retrieval["selected"]
    selected_layer = int(selected_linear["layer"])
    selected_view = str(selected_linear["view"])
    with np.load(
        output_dir(run_path) / "improvement" / "retrieval_sweep_projection.npz"
    ) as projection_data:
        base_values = project(
            views[selected_view][:, captured_layers.index(selected_layer)],
            projection_data["object_mean"].astype(np.float32),
            projection_data["object_basis"].astype(np.float32),
        )
    layers = [
        layer
        for layer in (
            improvement.get("local_layers") if local else improvement["layers"]
        )
        if int(layer) in captured_layers
    ]
    x, input_manifest = build_multiview_input(
        views,
        captured_layers,
        [int(layer) for layer in layers],
        per_view_dim=8 if local else 16,
        base_values=base_values,
    )
    train_indices = np.asarray(
        [index for index, row in enumerate(records) if row["split"] == "train"],
        dtype=int,
    )
    labels = build_labels(records)
    latent_dimensions = [
        int(value) for value in cfg["latent_dimensions"]
    ]
    hidden_dimensions = cfg.get(
        "hidden_dimensions", [cfg.get("hidden_dimension", 128)]
    )
    if not isinstance(hidden_dimensions, list):
        hidden_dimensions = [hidden_dimensions]
    if local:
        hidden_dimensions = hidden_dimensions[:1]
    epochs_cfg = cfg.get("epochs", 30)
    epoch_values = [
        int(value)
        for value in (
            epochs_cfg if isinstance(epochs_cfg, list) else [epochs_cfg]
        )
    ]
    variants = [str(value) for value in cfg["variants"]]
    results = []
    checkpoints: dict[
        tuple[str, int, int, int], dict[str, torch.Tensor]
    ] = {}
    tasks = [
        (variant, latent_dim, int(hidden_dim), epochs)
        for variant in variants
        for latent_dim in latent_dimensions
        for hidden_dim in hidden_dimensions
        for epochs in epoch_values
    ]
    for variant, latent_dim, hidden_dim, epochs in tqdm(
        tasks, desc="nonlinear encoder sweep", unit="model"
    ):
        base_dim = min(base_values.shape[1], latent_dim)
        model, losses = train_encoder(
            x=x,
            train_indices=train_indices,
            labels=labels,
            variant=variant,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            epochs=epochs,
            base_dim=base_dim,
            score_callback=lambda candidate: robust_validation_score(
                candidate,
                x,
                train_indices,
                selection_records,
            ),
        )
        z = encode_all(model, x)
        split_reports, _ = evaluate_by_split(
            train_vectors=z[train_indices],
            all_vectors=z,
            records=selection_records,
            train_indices=train_indices,
            splits=EVAL_SPLITS,
        )
        heldout_indices = np.asarray(
            [
                index
                for index, row in enumerate(records)
                if row["split"] == "heldout_vocab"
            ],
            dtype=int,
        )
        decoder = decode_representation(
            z[train_indices],
            z[heldout_indices],
            [records[index] for index in train_indices],
            [records[index] for index in heldout_indices],
        )
        row = {
            "variant": variant,
            "latent_dimension": latent_dim,
            "hidden_dimension": hidden_dim,
            "base_dimension": base_dim,
            "epochs": epochs,
            "selected_epoch": losses["selected_epoch"],
            "final_loss": losses["values"][-1],
            "retrieval": split_reports,
            "operation_macro_f1": decoder["operation"]["macro_f1"],
            "edit_macro_f1": decoder["edit_type"]["macro_f1"],
            "lexical_probe_accuracy": nuisance_probe_accuracy(
                z[train_indices],
                [
                    records[index]["surface"]["lexical_family"]
                    for index in train_indices
                ],
            ),
            "template_probe_accuracy": nuisance_probe_accuracy(
                z[train_indices],
                [
                    records[index]["surface"]["template_id"]
                    for index in train_indices
                ],
            ),
        }
        results.append(row)
        checkpoints[(variant, latent_dim, hidden_dim, epochs)] = {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        }
    selected = max(
        results,
        key=lambda row: (
            min(
                row["retrieval"]["validation"]["top1"],
                row["retrieval"]["template_validation"]["top1"],
            ),
            (
                row["retrieval"]["validation"]["top1"]
                + row["retrieval"]["template_validation"]["top1"]
            )
            / 2,
            -row["lexical_probe_accuracy"],
            row["operation_macro_f1"],
            -row["latent_dimension"],
        ),
    )
    key = (
        str(selected["variant"]),
        int(selected["latent_dimension"]),
        int(selected["hidden_dimension"]),
        int(selected["epochs"]),
    )
    out = output_dir(run_path) / "improvement"
    torch.save(
        {
            "state_dict": checkpoints[key],
            "input_dim": int(x.shape[1]),
            "base_dim": int(selected["base_dimension"]),
            "input_manifest": input_manifest,
            "selected": selected,
            "label_classes": labels.classes,
        },
        out / "nonlinear_encoder.pt",
    )
    report = {
        "local": local,
        "input_dim": int(x.shape[1]),
        "base_dimension": int(base_values.shape[1]),
        "input_manifest": input_manifest,
        "results": results,
        "selected": selected,
        "selection_rule": (
            "maximin vocabulary/template-validation retrieval, lower lexical "
            "probe, operation F1, lower dimension"
        ),
    }
    write_json(out / "nonlinear_sweep.json", report)
    return report


def build_multiview_input(
    views: dict[str, np.ndarray],
    captured_layers: list[int],
    layers: list[int],
    *,
    per_view_dim: int,
    base_values: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Concatenate deterministic random sketches of every layer/view."""
    features = []
    manifest = []
    for layer in layers:
        layer_col = captured_layers.index(layer)
        for view_name in sorted(views):
            values = views[view_name][:, layer_col]
            mean = values.mean(axis=0)
            basis = random_projection(
                values.shape[1],
                per_view_dim,
                seed=layer * 101 + sum(map(ord, view_name)),
            )
            features.append((values - mean) @ basis.T)
            manifest.append(
                {
                    "layer": layer,
                    "view": view_name,
                    "dimensions": per_view_dim,
                    "seed": layer * 101 + sum(map(ord, view_name)),
                }
            )
    sketches = np.concatenate(features, axis=1).astype(np.float32)
    scale = np.std(sketches, axis=0, keepdims=True)
    sketches = sketches / np.maximum(scale, 1e-5)
    if base_values is None:
        return sketches, manifest
    base = base_values.astype(np.float32)
    base /= np.maximum(np.linalg.norm(base, axis=1, keepdims=True), 1e-8)
    return np.concatenate([base, sketches], axis=1), manifest


def build_labels(records: list[dict[str, Any]]) -> LabelBundle:
    """Encode factorized object and nuisance labels."""
    raw = {
        "graph": [row["canonical_graph_id"] for row in records],
        "edit": [row["edit_type"] for row in records],
        "operation": [row["observed"]["operation"] for row in records],
        "value": [str(row["observed"].get("result")) for row in records],
        "role": [str(row["observed"].get("target")) for row in records],
        "next_edit": [EDIT_ORDER[row["edit_type"]] for row in records],
        "lexical": [row["surface"]["lexical_family"] for row in records],
        "template": [row["surface"]["template_id"] for row in records],
    }
    classes = {name: sorted(set(values)) for name, values in raw.items()}
    encoded = {
        name: np.asarray(
            [classes[name].index(value) for value in values], dtype=np.int64
        )
        for name, values in raw.items()
    }
    return LabelBundle(classes=classes, **encoded)


def train_encoder(
    *,
    x: np.ndarray,
    train_indices: np.ndarray,
    labels: LabelBundle,
    variant: str,
    latent_dim: int,
    hidden_dim: int,
    epochs: int,
    base_dim: int = 0,
    score_callback: Callable[[ObjectEncoder], tuple[float, ...]] | None = None,
) -> tuple[ObjectEncoder, dict[str, Any]]:
    """Train one frozen-input multitask encoder."""
    torch.manual_seed(42)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    class_counts = {
        name: len(values) for name, values in labels.classes.items()
    }
    model = ObjectEncoder(
        x.shape[1], hidden_dim, latent_dim, class_counts, base_dim=base_dim
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    features = torch.from_numpy(x).to(device)
    targets = {
        name: torch.from_numpy(getattr(labels, name)).to(device)
        for name in class_counts
    }
    index_tensor = torch.from_numpy(train_indices).to(device)
    losses = []
    best_score = score_callback(model) if score_callback else ()
    best_epoch = 0
    best_state = deepcopy(model.state_dict())
    for _epoch in range(epochs):
        model.train()
        order = index_tensor[torch.randperm(len(index_tensor), device=device)]
        batch_losses = []
        for start in range(0, len(order), 64):
            indices = order[start : start + 64]
            adversarial = 0.2 if variant == "contrastive_adversarial" else 0.0
            z, outputs = model(features[indices], adversarial_scale=adversarial)
            loss = (
                torch.nn.functional.cross_entropy(
                    outputs["graph"], targets["graph"][indices]
                )
                + 0.3
                * torch.nn.functional.cross_entropy(
                    outputs["edit"], targets["edit"][indices]
                )
                + 0.3
                * torch.nn.functional.cross_entropy(
                    outputs["operation"], targets["operation"][indices]
                )
                + 0.2
                * torch.nn.functional.cross_entropy(
                    outputs["value"], targets["value"][indices]
                )
                + 0.2
                * torch.nn.functional.cross_entropy(
                    outputs["next_edit"], targets["next_edit"][indices]
                )
                + 0.2
                * torch.nn.functional.cross_entropy(
                    outputs["role"], targets["role"][indices]
                )
            )
            if variant in {"contrastive", "contrastive_adversarial"}:
                loss = loss + 0.3 * supervised_contrastive_loss(
                    z, targets["graph"][indices]
                )
            if adversarial:
                loss = loss + 0.2 * (
                    torch.nn.functional.cross_entropy(
                        outputs["lexical"], targets["lexical"][indices]
                    )
                    + torch.nn.functional.cross_entropy(
                        outputs["template"], targets["template"][indices]
                    )
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(batch_losses)))
        epoch = _epoch + 1
        if score_callback and (epoch % 3 == 0 or epoch == epochs):
            score = score_callback(model)
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, {
        "values": losses,
        "selected_epoch": best_epoch,
        "selection_score": list(best_score),
    }


def robust_validation_score(
    model: ObjectEncoder,
    x: np.ndarray,
    train_indices: np.ndarray,
    records: list[dict[str, Any]],
) -> tuple[float, ...]:
    """Score checkpoints without consulting either final held-out split."""
    z = encode_all(model, x)
    reports, _ = evaluate_by_split(
        train_vectors=z[train_indices],
        all_vectors=z,
        records=records,
        train_indices=train_indices,
        splits=("validation", "template_validation"),
    )
    validation = reports["validation"]
    template = reports["template_validation"]
    return (
        min(validation["top1"], template["top1"]),
        (validation["top1"] + template["top1"]) / 2,
        validation["mean_retrieval_margin"] or -1e9,
    )


def supervised_contrastive_loss(
    z: torch.Tensor, labels: torch.Tensor, *, temperature: float = 0.1
) -> torch.Tensor:
    """Pull same-graph samples together while using all other rows as negatives."""
    logits = z @ z.T / temperature
    mask = labels[:, None] == labels[None, :]
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    positive = mask & ~eye
    logits = logits.masked_fill(eye, -1e9)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    counts = positive.sum(dim=1)
    valid = counts > 0
    if not torch.any(valid):
        return z.sum() * 0.0
    return -(
        (log_prob * positive).sum(dim=1)[valid] / counts[valid]
    ).mean()


def encode_all(model: ObjectEncoder, x: np.ndarray) -> np.ndarray:
    """Encode all records on the model device."""
    device = next(model.parameters()).device
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x), 256):
            z, _ = model(
                torch.from_numpy(x[start : start + 256]).to(device),
                adversarial_scale=0.0,
            )
            outputs.append(z.cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)
