"""Artifact-derived success checklist for the three paper replications."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.experiments.layer_replications.common import replication_dir
from src.runtime.artifact_store import write_json
from src.runtime.config import load_config


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _evaluation_path(run_path: Path, setting: str) -> Path:
    return (
        replication_dir(run_path)
        / "zhang_single_layer_rl/evaluations"
        / f"{setting}.json"
    )


def _check(
    label: str, expected: str, observed: Any | None, passed: bool | None
) -> dict[str, Any]:
    return {
        "label": label,
        "expected": expected,
        "observed": observed,
        "status": "pending" if passed is None else "passed" if passed else "failed",
    }


def _paper_status(checks: list[dict[str, Any]]) -> str:
    statuses = {row["status"] for row in checks}
    if "pending" in statuses:
        return "pending"
    return "passed" if statuses == {"passed"} else "failed"


def _lad_checks(run_path: Path) -> list[dict[str, Any]]:
    report = _load_json(replication_dir(run_path) / "lad_robustness/report.json")
    if report is None:
        return [
            _check("complete intervention matrix", "all drop/swap cells", None, None),
            _check(
                "middle layers are deletion-robust", "middle KL < edge KL", None, None
            ),
            _check("middle layers are swap-robust", "middle KL < edge KL", None, None),
            _check(
                "middle swaps are less harmful than drops",
                "swap KL < drop KL in middle half",
                None,
                None,
            ),
            _check(
                "middle deletion preserves predictions",
                "middle top-1 consistency > edge consistency",
                None,
                None,
            ),
        ]
    curves = report["curves"]
    by_kind = {
        kind: sorted(
            (row for row in curves if row["intervention"] == kind),
            key=lambda row: row["layer"],
        )
        for kind in ("drop", "swap")
    }

    def bands(kind: str, metric: str) -> tuple[float, float]:
        rows = by_kind[kind]
        width = max(int(row["layer"]) for row in rows) + 1
        middle = [
            float(row[metric])
            for row in rows
            if width // 4 <= int(row["layer"]) < 3 * width // 4
        ]
        edges = [
            float(row[metric])
            for row in rows
            if int(row["layer"]) < width // 4 or int(row["layer"]) >= 3 * width // 4
        ]
        return float(np.mean(middle)), float(np.mean(edges))

    drop_middle, drop_edges = bands("drop", "kl")
    swap_middle, swap_edges = bands("swap", "kl")
    middle_drop_top1, edge_drop_top1 = bands("drop", "top1_consistency")
    return [
        _check(
            "complete intervention matrix",
            "all configured drop/swap cells",
            int(report["tasks"]),
            bool(report.get("complete")),
        ),
        _check(
            "middle layers are deletion-robust",
            "middle KL < edge KL",
            {"middle": drop_middle, "edge": drop_edges},
            drop_middle < drop_edges,
        ),
        _check(
            "middle layers are swap-robust",
            "middle KL < edge KL",
            {"middle": swap_middle, "edge": swap_edges},
            swap_middle < swap_edges,
        ),
        _check(
            "middle swaps are less harmful than drops",
            "swap KL < drop KL in middle half",
            {"swap": swap_middle, "drop": drop_middle},
            swap_middle < drop_middle,
        ),
        _check(
            "middle deletion preserves predictions",
            "middle top-1 consistency > edge consistency",
            {"middle": middle_drop_top1, "edge": edge_drop_top1},
            middle_drop_top1 > edge_drop_top1,
        ),
    ]


def _yang_layer_centers(run_path: Path) -> dict[str, float]:
    path = replication_dir(run_path) / "yang_symbolic/head_scores.csv"
    if not path.exists():
        return {}
    weighted: dict[str, list[tuple[float, float]]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["significant_fwer_0_05"].lower() != "true":
                continue
            weighted.setdefault(row["mechanism"], []).append(
                (float(row["layer"]), float(row["mean_causal_score"]))
            )
    return {
        mechanism: float(
            np.average(
                [layer for layer, _score in rows],
                weights=[max(score, 0.0) for _layer, score in rows],
            )
        )
        for mechanism, rows in weighted.items()
        if rows and any(score > 0 for _layer, score in rows)
    }


def _yang_checks(run_path: Path) -> list[dict[str, Any]]:
    report = _load_json(replication_dir(run_path) / "yang_symbolic/report.json")
    selection = _load_json(
        replication_dir(run_path) / "yang_symbolic/selection_report.json"
    )
    if report is None:
        return [
            _check(
                "complete causal-mediation matrix", "all selected pairs", None, None
            ),
            _check(
                "significant heads for all three mechanisms",
                "FWER p<0.05 count > 0 for each mechanism",
                None,
                None,
            ),
            _check(
                "three-stage depth hierarchy",
                "abstraction center < induction center < retrieval center",
                None,
                None,
            ),
        ]
    counts = {
        name: int(values["significant_heads"])
        for name, values in report["mechanisms"].items()
    }
    centers = _yang_layer_centers(run_path)
    ordered = all(
        name in centers
        for name in ("symbol_abstraction", "symbolic_induction", "retrieval")
    )
    if ordered:
        ordered = (
            centers["symbol_abstraction"]
            < centers["symbolic_induction"]
            < centers["retrieval"]
        )
    return [
        _check(
            "clean-correct paper-sized selection",
            "100 pairs per context/rule cell",
            selection,
            selection is not None and int(selection.get("selected", 0)) == 400,
        ),
        _check(
            "significant heads for all three mechanisms",
            "FWER p<0.05 count > 0 for each mechanism",
            counts,
            all(count > 0 for count in counts.values()) and len(counts) == 3,
        ),
        _check(
            "three-stage depth hierarchy",
            "abstraction center < induction center < retrieval center",
            centers or None,
            ordered if centers else None,
        ),
    ]


def _zhang_checks(run_path: Path) -> list[dict[str, Any]]:
    config = load_config(run_path)
    layers = tuple(
        int(value) for value in config["single_layer_rl"]["core_scan_layers"]
    )
    reports = {
        name: _load_json(_evaluation_path(run_path, name))
        for name in ("base", "full", *(f"layer-{layer:02d}" for layer in layers))
    }
    missing = [name for name, report in reports.items() if report is None]
    if missing:
        return [
            _check(
                "core base/full/layer evaluations",
                "base, full, and layers 1/7/10/12/24",
                {"missing": missing},
                None,
            ),
            _check(
                "full GRPO improves the base model",
                "full math average > base",
                None,
                None,
            ),
            _check(
                "best published middle layer matches full GRPO",
                "max contribution at layers 10/12 >= 0.9",
                None,
                None,
            ),
            _check(
                "middle-layer contribution concentration",
                "mean layers 7/10/12 > mean layers 1/24",
                None,
                None,
            ),
            _check(
                "late control underperforms best middle layer",
                "layer 24 contribution < max layer 10/12 contribution",
                None,
                None,
            ),
        ]
    scores = {name: float(report["math_average"]) for name, report in reports.items()}
    gain = scores["full"] - scores["base"]
    contributions = (
        {
            layer: (scores[f"layer-{layer:02d}"] - scores["base"]) / gain
            for layer in layers
        }
        if gain > 0
        else {}
    )
    middle = [contributions[layer] for layer in (7, 10, 12)]
    edges = [contributions[layer] for layer in (1, 24)]
    return [
        _check(
            "full GRPO improves the base model",
            "full math average > base",
            {"base": scores["base"], "full": scores["full"]},
            gain > 0,
        ),
        _check(
            "best published middle layer matches full GRPO",
            "max contribution at layers 10/12 >= 0.9",
            {str(layer): contributions[layer] for layer in (10, 12)}
            if contributions
            else None,
            max(contributions[10], contributions[12]) >= 0.9
            if contributions
            else False,
        ),
        _check(
            "middle-layer contribution concentration",
            "mean layers 7/10/12 > mean layers 1/24",
            {"middle": float(np.mean(middle)), "edge": float(np.mean(edges))}
            if contributions
            else None,
            float(np.mean(middle)) > float(np.mean(edges)) if contributions else False,
        ),
        _check(
            "late control underperforms best middle layer",
            "layer 24 contribution < max layer 10/12 contribution",
            {str(layer): contributions[layer] for layer in (10, 12, 24)}
            if contributions
            else None,
            contributions[24] < max(contributions[10], contributions[12])
            if contributions
            else False,
        ),
    ]


def build(
    lad_run: Path,
    yang_run: Path,
    zhang_run: Path,
    *,
    output: Path,
) -> dict[str, Any]:
    """Write JSON and Markdown reports derived only from completed artifacts."""
    papers = [
        {
            "key": "lad",
            "paper": "The Remarkable Robustness of LLMs: Stages of Inference?",
            "fidelity": "paper protocol on a paper-listed Qwen2.5-1.5B model",
            "checks": _lad_checks(lad_run),
        },
        {
            "key": "yang",
            "paper": "Emergent Symbolic Mechanisms Support Abstract Reasoning in LLMs",
            "fidelity": "paper-sized Qwen2.5-7B identity-rule CMA",
            "checks": _yang_checks(yang_run),
        },
        {
            "key": "zhang",
            "paper": (
                "Is One Layer Enough? Training A Single Transformer Layer Can "
                "Match Full-Parameter RL Training"
            ),
            "fidelity": "published Qwen3-1.7B anchor layers; full 28-layer scan optional",
            "checks": _zhang_checks(zhang_run),
        },
    ]
    for paper in papers:
        paper["status"] = _paper_status(paper["checks"])
    result = {
        "status": _paper_status([{"status": paper["status"]} for paper in papers]),
        "papers": papers,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output.with_suffix(".json"), result)
    lines = [
        "# Layer-paper replication checklist",
        "",
        "Generated from pulled artifacts. A completed job passes only when its",
        "central empirical result also passes the paper-grounded check.",
        "",
    ]
    symbols = {"passed": "[x]", "failed": "[!]", "pending": "[ ]"}
    for paper in papers:
        lines.extend(
            [
                f"## {symbols[paper['status']]} {paper['paper']}",
                "",
                f"Fidelity: {paper['fidelity']}.",
                "",
            ]
        )
        for row in paper["checks"]:
            lines.append(
                f"- {symbols[row['status']]} {row['label']}: "
                f"expected {row['expected']}; observed "
                f"{json.dumps(row['observed'], sort_keys=True)}"
            )
        lines.append("")
    output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    return result
