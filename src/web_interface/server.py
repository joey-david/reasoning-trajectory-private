from __future__ import annotations

import csv
import json
import re
import threading
import traceback
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import yaml

from src.analysis.common import activation_layers, load_generations
from src.analysis.tools import TOOLS, run_all_tools, run_tool, tool_specs
from src.config import load_run_config, run_path
from src.data import load_samples
from src.env import load_dotenv
from src.generation import run_generation


ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).resolve().parent / "static"
BUILT_IN_MODELS = [
    {"id": "Qwen/Qwen2.5-0.5B-Instruct", "label": "Qwen2.5 0.5B Instruct", "backend": "hf"},
    {"id": "Qwen/Qwen2.5-1.5B-Instruct", "label": "Qwen2.5 1.5B Instruct", "backend": "hf"},
    {"id": "mlx-community/Qwen2.5-0.5B-Instruct-4bit", "label": "Apple MLX Qwen2.5 0.5B 4-bit", "backend": "mlx"},
    {"id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "label": "DeepSeek R1 Distill Llama 8B", "backend": "hf"},
]


@dataclass
class Job:
    id: str
    label: str
    status: str = "queued"
    outputs: list[str] = field(default_factory=list)
    error: str = ""


JOBS: dict[str, Job] = {}
JOB_KEYS: dict[str, str] = {}


def run(host: str = "127.0.0.1", port: int = 8765) -> None:
    load_dotenv(ROOT)
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"http://{host}:{port}")
    server.serve_forever()


class Handler(BaseHTTPRequestHandler):
    server_version = "ReasoningTrajectoryWeb/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                return self._static("index.html")
            if parsed.path.startswith("/assets/"):
                return self._static(parsed.path.removeprefix("/assets/"))
            if parsed.path == "/api/state":
                return self._json(state())
            if parsed.path == "/api/run":
                query = parse_qs(parsed.query)
                return self._json(run_state(_required(query, "run_path")))
            if parsed.path == "/api/tool-data":
                query = parse_qs(parsed.query)
                return self._json(tool_data(_required(query, "run_path"), _required(query, "tool"), query))
            if parsed.path.startswith("/api/jobs/"):
                job_id = parsed.path.rsplit("/", 1)[-1]
                return self._json(JOBS.get(job_id, Job(job_id, "missing", "missing")).__dict__)
            self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            body = self._read_json()
            if parsed.path == "/api/generate":
                return self._json(start_generation(body), HTTPStatus.ACCEPTED)
            if parsed.path == "/api/analyze":
                return self._json(start_analysis(body), HTTPStatus.ACCEPTED)
            self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _static(self, name: str) -> None:
        path = (STATIC / unquote(name)).resolve()
        if not path.is_file() or STATIC not in path.parents:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = "text/html" if path.suffix == ".html" else "text/css" if path.suffix == ".css" else "application/javascript"
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def state() -> dict[str, Any]:
    return {
        "models": BUILT_IN_MODELS,
        "datasets": datasets(),
        "runs": runs(),
        "tools": tool_specs(),
    }


def datasets() -> list[dict[str, str]]:
    items = []
    for path in sorted((ROOT / "datasets").glob("*.json*")):
        items.append({"path": str(path.relative_to(ROOT)), "label": path.name})
    return items


def runs() -> list[dict[str, Any]]:
    items = []
    for config in sorted((ROOT / "runs").glob("*/*/config.yaml")):
        cfg = load_run_config(config)
        items.append({
            "path": str(config.parent.relative_to(ROOT)),
            "model_name": cfg.get("model_name", config.parents[1].name),
            "name": config.parent.name,
            "has_generation": (config.parent / "generation" / "generations.jsonl").exists(),
        })
    return items


def run_state(run_path_text: str) -> dict[str, Any]:
    cfg = load_run_config(ROOT / run_path_text)
    base = run_path(cfg)
    rows = load_generations(cfg)
    return {
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "run_path": str(base.relative_to(ROOT)),
        "samples": len(load_samples(cfg)),
        "generations": len(rows),
        "layers": activation_layers(cfg),
        "analyses": sorted(path.name for path in (base / "analysis").glob("*")) if (base / "analysis").exists() else [],
    }


def start_generation(body: dict[str, Any]) -> dict[str, Any]:
    cfg_path = _ensure_config(body)
    return _start_job(f"generate {cfg_path.parent.name}", lambda: [str(run_generation(load_run_config(cfg_path)).relative_to(ROOT))])


def start_analysis(body: dict[str, Any]) -> dict[str, Any]:
    cfg = load_run_config(ROOT / body["run_path"])
    name = body.get("tool", "all")
    params = body.get("params") or {}
    if name == "all":
        return _start_job("analyze all", lambda: [str(path.relative_to(ROOT)) for path in run_all_tools(cfg, params)])
    if name not in TOOLS:
        raise ValueError(f"Unknown tool: {name}")
    tool_params = params.get(name, {})
    return _start_job(f"analyze {name}", lambda: [str(run_tool(cfg, name, tool_params).relative_to(ROOT))])


def tool_data(run_path_text: str, tool: str, query: dict[str, list[str]]) -> dict[str, Any]:
    cfg = load_run_config(ROOT / run_path_text)
    params = _params_for_tool(cfg, tool, query)
    target = _tool_output_path(cfg, tool, params)

    if target.exists():
        data = _load_tool_output(tool, target)
        if isinstance(data, dict):
            data.setdefault("status", "done")
            data.setdefault("path", str(target.relative_to(ROOT)))
        return data

    job = _start_analysis_once(cfg, tool, params, target)
    return {
        "status": job.status,
        "tool": tool,
        "job": job.__dict__,
        "path": str(target.relative_to(ROOT)),
        "message": f"Missing analysis output; started {tool}.",
    }


def _params_for_tool(config: dict[str, Any], tool: str, query: dict[str, list[str]]) -> dict[str, Any]:
    if tool == "generation_summary":
        return {}
    if tool == "activation_norms":
        return {}
    if tool == "trajectory_projection":
        return {
            "layer": _optional(query, "layer") or _last_layer(config),
            "interval": int(_optional(query, "interval") or 16),
            "method": _optional(query, "method") or "pca",
            "max_points": int(_optional(query, "max_points") or 12000),
        }
    if tool == "pca_components":
        return {
            "layer": _optional(query, "layer") or _last_layer(config),
            "n": int(_optional(query, "n") or 24),
            "max_vectors": int(_optional(query, "max_vectors") or 20000),
        }
    raise ValueError(f"Unknown tool: {tool}")


def _tool_output_path(config: dict[str, Any], tool: str, params: dict[str, Any]) -> Path:
    base = run_path(config)
    if tool == "generation_summary":
        return base / "analysis" / "generation_summary.csv"
    if tool == "activation_norms":
        return base / "analysis" / "activation_norms.csv"
    if tool == "trajectory_projection":
        return base / "analysis" / f"trajectory_projection_layer{params['layer']}_i{int(params['interval'])}_{params['method']}.json"
    if tool == "pca_components":
        return base / "analysis" / f"pca_components_layer{params['layer']}_n{int(params['n'])}.json"
    raise ValueError(f"Unknown tool: {tool}")


def _load_tool_output(tool: str, path: Path) -> dict[str, Any]:
    if tool == "generation_summary":
        return {"status": "done", "tool": tool, "rows": _read_csv(path)}
    if tool == "activation_norms":
        return {"status": "done", "tool": tool, "rows": _read_csv(path)}
    return _read_json_file(path)


def _start_analysis_once(config: dict[str, Any], tool: str, params: dict[str, Any], target: Path) -> Job:
    key = f"{run_path(config)}::{tool}::{json.dumps(params, sort_keys=True)}"
    existing = JOB_KEYS.get(key)
    if existing and existing in JOBS and JOBS[existing].status in {"queued", "running"}:
        return JOBS[existing]

    job = _start_job(
        f"analyze {tool}",
        lambda: [str(run_tool(config, tool, params).relative_to(ROOT))],
    )
    JOB_KEYS[key] = job["id"]
    return JOBS[job["id"]]


def _ensure_config(body: dict[str, Any]) -> Path:
    if body.get("run_path"):
        return ROOT / body["run_path"] / "config.yaml"
    model_name = body["model_name"]
    dataset_path = body["dataset_path"]
    run_name = _slug(body.get("run_name") or "web_run")
    model_slug = _slug(model_name.rsplit("/", 1)[-1])
    path = ROOT / "runs" / model_slug / run_name / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "model_name": model_name,
        "backend": body.get("backend", "hf"),
        "dataset_path": dataset_path,
        "device_map": body.get("device_map", "auto"),
        "torch_dtype": body.get("torch_dtype", "auto"),
        "trust_remote_code": bool(body.get("trust_remote_code", False)),
        "max_new_tokens": int(body.get("max_new_tokens", 160)),
        "seeds": [int(value) for value in body.get("seeds", [0])],
        "temperatures": [float(value) for value in body.get("temperatures", [0.7])],
        "layers": [int(value) for value in body.get("layers", [0, 4, 8])],
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _start_job(label: str, action) -> dict[str, Any]:
    job_id = f"job-{len(JOBS) + 1}"
    job = Job(job_id, label)
    JOBS[job_id] = job

    def work() -> None:
        job.status = "running"
        try:
            job.outputs = action()
            job.status = "done"
        except Exception as exc:
            job.error = f"{exc}\n{traceback.format_exc()}"
            job.status = "error"

    threading.Thread(target=work, daemon=True).start()
    return job.__dict__


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing analysis output: {path.relative_to(ROOT)}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing analysis output: {path.relative_to(ROOT)}")
    return json.loads(path.read_text(encoding="utf-8"))


def _last_layer(config: dict[str, Any]) -> str:
    layers = [str(layer) for layer in config.get("layers", [])]
    return layers[-1] if layers else "0"


def _required(query: dict[str, list[str]], key: str) -> str:
    value = _optional(query, key)
    if value is None:
        raise ValueError(f"Missing query parameter: {key}")
    return value


def _optional(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    return values[0] if values else None


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "run"
