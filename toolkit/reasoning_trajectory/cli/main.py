from __future__ import annotations

import argparse
import json
from pathlib import Path

from reasoning_trajectory.analysis import analyze_run, export_compression, export_failure_reports, export_run_report, run_config
from reasoning_trajectory.branching import export_basins
from reasoning_trajectory.core.converters import convert_legacy_jsonl
from reasoning_trajectory.core.registry import list_tools
from reasoning_trajectory.core.storage import save_jsonl
from reasoning_trajectory.core.utils import dependency_status
from reasoning_trajectory.dashboard.app import launch_or_export_dashboard
from reasoning_trajectory.extract.generations import extract_from_config
from reasoning_trajectory.extract.hf_check import hf_inference_check
from reasoning_trajectory.extract.token_steps import parse_steps
from reasoning_trajectory.metrics.alignment import export_alignment
from reasoning_trajectory.metrics.geometry import export_geometry
from reasoning_trajectory.verifiers.lean import LeanVerifier
from reasoning_trajectory.verifiers.python_tests import verify_python_file
from reasoning_trajectory.verifiers.smt import SMTVerifier
from reasoning_trajectory.verifiers.symbolic_math import verify_symbolic
from reasoning_trajectory.visualize.trajectory_3d import export_trajectory_plot


def add_common(parser):
    parser.add_argument("--config")
    parser.add_argument("--input")
    parser.add_argument("--out")
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rt")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run"); add_common(p); p.add_argument("--layer"); p.add_argument("--no-analyze", action="store_true")
    p = sub.add_parser("analyze"); add_common(p); p.add_argument("--layer"); p.add_argument("--dims", type=int, default=2)
    p = sub.add_parser("extract"); add_common(p); p.add_argument("extract_cmd", nargs="?", choices=["run", "hf-check"])
    p = sub.add_parser("metrics"); add_common(p); p.add_argument("--layer")
    p = sub.add_parser("plot"); add_common(p); p.add_argument("--layer"); p.add_argument("--color-by", default="correctness"); p.add_argument("--method", default="pca")
    p = sub.add_parser("dashboard"); add_common(p)
    p = sub.add_parser("compression"); add_common(p); p.add_argument("--layer"); p.add_argument("--dims", type=int, default=2)
    p = sub.add_parser("basins"); add_common(p); p.add_argument("--layer"); p.add_argument("--clusters", type=int, default=3)
    p = sub.add_parser("failures"); add_common(p); p.add_argument("--layer")
    p = sub.add_parser("report"); add_common(p)
    p = sub.add_parser("parse-steps"); add_common(p)
    p = sub.add_parser("convert-legacy"); add_common(p)

    verify = sub.add_parser("verify")
    vsub = verify.add_subparsers(dest="verify_cmd", required=True)
    p = vsub.add_parser("python"); add_common(p); p.add_argument("--tests", required=True)
    p = vsub.add_parser("symbolic"); add_common(p); p.add_argument("--expr", required=True); p.add_argument("--expected", required=True)
    p = vsub.add_parser("lean"); add_common(p)
    p = vsub.add_parser("smt"); add_common(p)

    sub.add_parser("list-tools")
    sub.add_parser("doctor")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "run":
        print(run_config(args.config, args.out, analyze=not args.no_analyze, layer=args.layer))
    elif args.cmd == "analyze":
        result = analyze_run(args.input, args.layer, args.dims)
        print(json.dumps({k: str(v) if v is not None else None for k, v in result.items()}, indent=2))
    elif args.cmd == "extract":
        if args.extract_cmd == "hf-check":
            print(json.dumps(hf_inference_check(args.config, args.out), indent=2))
        else:
            extract_from_config(args.config, args.out)
    elif args.cmd == "metrics":
        export_geometry(args.input, Path(args.out) / "geometry.jsonl", args.layer)
        export_alignment(args.input, Path(args.out) / "alignment.jsonl", args.layer)
    elif args.cmd == "plot":
        export_trajectory_plot(args.input, args.out, args.layer, args.color_by, args.method)
    elif args.cmd == "dashboard":
        launch_or_export_dashboard(args.input, args.out)
    elif args.cmd == "compression":
        export_compression(args.input, args.out, args.dims, args.layer)
    elif args.cmd == "basins":
        print(json.dumps(export_basins(args.input, args.out, args.clusters, args.layer), indent=2))
    elif args.cmd == "failures":
        export_failure_reports(args.input, args.out, args.layer)
    elif args.cmd == "report":
        print(export_run_report(args.input, args.out))
    elif args.cmd == "parse-steps":
        text = Path(args.input).read_text(encoding="utf-8")
        _write_rows([span.__dict__ for span in parse_steps(text)], args.out)
    elif args.cmd == "convert-legacy":
        save_jsonl(convert_legacy_jsonl(args.input), args.out)
    elif args.cmd == "verify":
        _verify(args)
    elif args.cmd == "list-tools":
        for spec in list_tools():
            print(f"{spec.name}\t{spec.category}\t{spec.cli}\t{spec.doc}")
    elif args.cmd == "doctor":
        print(json.dumps(dependency_status(["numpy", "sklearn", "plotly", "streamlit", "torch", "transformers", "sympy", "z3", "umap"]), indent=2))
    return 0


def _verify(args) -> None:
    if args.verify_cmd == "python":
        result = verify_python_file(args.input, args.tests)
    elif args.verify_cmd == "symbolic":
        result = verify_symbolic(args.expr, args.expected)
    elif args.verify_cmd == "lean":
        result = LeanVerifier().verify(Path(args.input).read_text(encoding="utf-8"))
    elif args.verify_cmd == "smt":
        result = SMTVerifier().verify(Path(args.input).read_text(encoding="utf-8"))
    print(json.dumps(result.__dict__, indent=2))


def _write_rows(rows: list[dict], path: str | None) -> None:
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    raise SystemExit(main())
