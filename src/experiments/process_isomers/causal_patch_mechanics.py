"""Activation-space patch construction and H3 artifact validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch

from src.models.introspection import get_decoder_layers, resolve_layer_indices
from src.runtime.data import load_samples

PATCH_MODES = ("full", "subspace")


@dataclass(slots=True)
class ProjectionSubspace:
    """Hold a validated linear map and its Moore-Penrose inverse."""

    path: Path
    weight: torch.Tensor
    pseudoinverse: torch.Tensor
    rank: int
    condition_number: float

    def swap(
        self,
        *,
        target: torch.Tensor,
        donor: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Replace target row-space coordinates with donor coordinates.

        Args:
            target: Target value or index.
            donor: Donor tensor or trace supplying the patched activation.

        Returns:
            The computed aligned values described above.
        """
        target = target.float().cpu()
        donor = donor.float().cpu()
        full_delta = donor - target
        coordinate_delta = self.weight @ full_delta
        subspace_delta = self.pseudoinverse @ coordinate_delta
        reconstructed = target + subspace_delta
        coordinate_residual = self.weight @ reconstructed - self.weight @ donor
        projected_delta = self.pseudoinverse @ (self.weight @ subspace_delta)
        orthogonal_leakage = subspace_delta - projected_delta
        return reconstructed, {
            "full_delta_norm": float(torch.linalg.vector_norm(full_delta)),
            "subspace_delta_norm": float(torch.linalg.vector_norm(subspace_delta)),
            "retained_delta_fraction": float(
                torch.linalg.vector_norm(subspace_delta)
                / torch.linalg.vector_norm(full_delta).clamp_min(1e-8)
            ),
            "coordinate_reconstruction_relative_residual": float(
                torch.linalg.vector_norm(coordinate_residual)
                / torch.linalg.vector_norm(self.weight @ donor).clamp_min(1e-8)
            ),
            "orthogonal_leakage_relative_residual": float(
                torch.linalg.vector_norm(orthogonal_leakage)
                / torch.linalg.vector_norm(subspace_delta).clamp_min(1e-8)
            ),
        }


def resolve_patch_modes(
    patch_cfg: dict[str, Any],
    override: str | None,
) -> tuple[str, ...]:
    """Resolve and validate the requested full or subspace patch modes.

    Args:
        patch_cfg: Causal patching configuration.
        override: Optional command-line patch-mode override.

    Returns:
        The computed aligned values described above.
    """
    if override and override != "both":
        modes = (override,)
    elif override == "both":
        modes = PATCH_MODES
    else:
        modes = tuple(str(mode) for mode in patch_cfg.get("patch_modes", PATCH_MODES))
    unknown = set(modes) - set(PATCH_MODES)
    if unknown:
        raise ValueError(f"Unsupported patch modes: {sorted(unknown)}")
    return modes


def resolve_patch_target(patch_cfg: dict[str, Any]) -> tuple[str, int]:
    """Resolve the configured component and layer for patching.

    Args:
        patch_cfg: Causal patching configuration.

    Returns:
        The computed aligned values described above.
    """
    if patch_cfg.get("alignment", "symbolic_step_end") != "symbolic_step_end":
        raise ValueError("H3 patch alignment must be symbolic_step_end")
    component = patch_cfg.get("component")
    layer = patch_cfg.get("layer")
    if component != "auto" and layer != "auto":
        return validate_patch_target(str(component), int(layer))
    report = json.loads(Path(patch_cfg["component_report"]).read_text())
    target = report.get("recommended_patch_target")
    if not target:
        raise ValueError("Component localization has no recommended patch target")
    if target.get("alignment") != "symbolic_step_end":
        raise ValueError("Component report does not authorize completed-step alignment")
    return validate_patch_target(str(target["component"]), int(target["layer"]))


def validate_patch_target(component: str, layer: int) -> tuple[str, int]:
    """Validate and return a supported component-layer target.

    Args:
        component: Activation component name.
        layer: Model layer index.

    Returns:
        The computed aligned values described above.
    """
    if component not in {"mlp_output", "attention_output"}:
        raise ValueError(f"Unsupported H3 patch component: {component!r}")
    return component, layer


def load_projection_subspace(
    path: Path,
    *,
    component: str,
    layer: int,
) -> ProjectionSubspace:
    """Load and validate a component-matched linear projection.

    Args:
        path: Filesystem path to read from or write to.
        component: Activation component name.
        layer: Model layer index.

    Returns:
        The validated projection basis and its metadata.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("component") != component:
        raise ValueError(
            f"Projection component {checkpoint.get('component')!r} does not match "
            f"patch component {component!r}"
        )
    if int(checkpoint.get("layer")) != layer:
        raise ValueError(
            f"Projection layer {checkpoint.get('layer')} does not match {layer}"
        )
    weight = checkpoint["weight"].float().cpu()
    if weight.ndim != 2 or weight.shape[1] != int(checkpoint["input_dim"]):
        raise ValueError(f"Malformed projection weight: {tuple(weight.shape)}")
    rank = int(torch.linalg.matrix_rank(weight).item())
    if rank != weight.shape[0]:
        raise ValueError(
            f"Projection is row-rank deficient: rank {rank}/{weight.shape[0]}"
        )
    singular_values = torch.linalg.svdvals(weight)
    condition_number = float(
        singular_values.max() / singular_values.clamp_min(1e-12).min()
    )
    return ProjectionSubspace(
        path=path,
        weight=weight,
        pseudoinverse=torch.linalg.pinv(weight),
        rank=rank,
        condition_number=condition_number,
    )


def completion_state_index(
    point: dict[str, Any],
    row: dict[str, Any],
) -> int:
    """Derive the fully resolved state after the symbolic interval's final token.

    Args:
        point: Boundary intervention specification.
        row: Generation or analysis record to process.

    Returns:
        The computed index, count, or status code.
    """
    token_count = len(row["generated_token_ids"])
    if token_count < 2:
        raise ValueError("Cannot patch a trace with fewer than two generated tokens")
    token_end = int(point["token_end"])
    if not 0 <= token_end < token_count - 1:
        raise ValueError(
            f"token_end {token_end} has no captured completion state "
            f"in a {token_count}-token trace"
        )
    return token_end + 1


def validate_pair_rows(
    pairs: list[dict[str, Any]],
    rows: dict[tuple[str, int], dict[str, Any]],
) -> None:
    """Validate pair diversity and captured completion states.

    Args:
        pairs: Matched treatment/control or process-isomer pairs.
        rows: Generation or analysis records to process.

    Returns:
        None.
    """
    for pair in pairs:
        if not pair.get("path_evidence"):
            raise ValueError(f"Pair {pair['pair_id']} lacks path-diversity evidence")
        if (
            pair["path_evidence"]["donor_history_hash"]
            == pair["path_evidence"]["target_history_hash"]
        ):
            raise ValueError(f"Pair {pair['pair_id']} has identical path hashes")
        for side in ("donor", "target"):
            point = pair[side]
            key = (str(point["sample_id"]), int(point["seed"]))
            if key not in rows:
                raise ValueError(f"Pair {pair['pair_id']} is missing {side} row {key}")
            completion_state_index(point, rows[key])


def load_completed_patches(
    path: Path,
) -> set[tuple[int, str, str, int]]:
    """Load the complete cell keys already persisted for resumption.

    Args:
        path: Filesystem path to read from or write to.

    Returns:
        The resulting unique values.
    """
    if not path.exists():
        return set()
    rows = load_samples(path.resolve())
    if any("patch_mode" not in row for row in rows):
        raise ValueError(
            f"{path} contains legacy rows without patch_mode; move or remove it "
            "before running the two-variant protocol"
        )
    return {
        (
            int(row["pair_id"]),
            str(row["patch_mode"]),
            str(row["condition"]),
            int(row["continuation"]),
        )
        for row in rows
    }


def output_degeneration_reasons(token_ids: list[int], text: str) -> list[str]:
    """Detect conservative, deterministic output-collapse signatures.

    Args:
        token_ids: Generated continuation token IDs.
        text: Generated text to inspect.

    Returns:
        The resulting ordered records or values.
    """
    reasons = []
    if not text.strip():
        reasons.append("empty_output")
    if len(token_ids) >= 32 and longest_identical_run(token_ids) >= 32:
        reasons.append("repeated_token_run")
    if len(token_ids) >= 100:
        unique_ratio = len(set(token_ids)) / len(token_ids)
        if unique_ratio < 0.02:
            reasons.append("very_low_token_diversity")
        ngrams = [
            tuple(token_ids[index : index + 4]) for index in range(len(token_ids) - 3)
        ]
        if ngrams and len(set(ngrams)) / len(ngrams) < 0.05:
            reasons.append("repeated_four_grams")
    return reasons


def longest_identical_run(values: list[int]) -> int:
    """Return the longest contiguous run of one token ID.

    Args:
        values: Values to summarize or transform.

    Returns:
        The computed index, count, or status code.
    """
    longest = 0
    current = 0
    previous = None
    for value in values:
        current = current + 1 if value == previous else 1
        longest = max(longest, current)
        previous = value
    return longest


class FirstForwardComponentPatch:
    """Replace one component's final prefill-token output exactly once."""

    def __init__(
        self,
        *,
        model: Any,
        layer: int,
        component: str,
        vector: torch.Tensor,
        expected_sequence_length: int,
    ) -> None:
        """Resolve the target module and retain the one-shot patch state.

        Args:
            model: Loaded model used for inference or transformation.
            layer: Model layer index.
            component: Activation component name.
            vector: Activation vector injected by the hook.
            expected_sequence_length: Sequence length on which the hook should fire.

        Returns:
            None.
        """
        decoder_layers = get_decoder_layers(model)
        resolved = resolve_layer_indices([layer], len(decoder_layers))[0]
        attribute = "mlp" if component == "mlp_output" else "self_attn"
        self.module = getattr(decoder_layers[resolved], attribute)
        self.vector = vector
        self.expected_sequence_length = expected_sequence_length
        self.handle = None
        self.applied = False

    def __enter__(self) -> FirstForwardComponentPatch:
        """Register the component hook for the next model forward pass.

        Args:
            None.

        Returns:
            This active context manager.
        """
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def _hook(self, _module, _inputs, output):
        """Replace the final prefill position once and preserve output shape.

        Args:
            _module: Hooked PyTorch module; unused by the callback.
            _inputs: Hook input tuple; unused by the callback.
            output: Output produced by the hooked module.

        Returns:
            A forward-hook callback that captures the requested output.
        """
        if self.applied:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3 or hidden.shape[1] != self.expected_sequence_length:
            raise RuntimeError(
                "Patch must occur on the full target prefill ending at token_end; "
                f"expected sequence length {self.expected_sequence_length}, "
                f"received {tuple(hidden.shape)}"
            )
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
        """Remove the registered hook when generation leaves the context.

        Args:
            exc_type: Exception type raised inside the context, if any.
            exc: Exception raised inside the context, if any.
            tb: Traceback associated with the exception, if any.

        Returns:
            None.
        """
        if self.handle is not None:
            self.handle.remove()
