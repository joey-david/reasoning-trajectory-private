"""Dynamically distribute rollouts to one persistent SSH worker per selected GPU."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from queue import Empty, Queue
import re
import shlex
import subprocess
import sys
from threading import Event, RLock
import time
import traceback
from typing import Any, TextIO

from tqdm.auto import tqdm

from src.runtime.config import RunConfig, load_config
from src.datasets.loaders import load_run_samples
from src.models.generation_pipeline import (
    generate_task,
    generation_key_for,
    sample_id_from_sample,
)
from src.models.hf_loader import load_hf_model_and_tokenizer
from src.runtime.run_io import load_generation_index


PREFIX = "@@ORCHESTRATOR@@"


def emit(message: dict[str, Any]) -> None:
    """Write one worker control message."""
    print(PREFIX + json.dumps(message, ensure_ascii=False), flush=True)


class RemoteProgress:
    """Forward generation progress while retaining speed during capture."""

    def __init__(self) -> None:
        self.last_speed: float | None = None

    def set_description(self, description: str, **_kwargs: Any) -> None:
        """Forward a tqdm-compatible progress description."""
        match = re.search(r"(\d+(?:\.\d+)?) tok/s", description)
        if match:
            self.last_speed = float(match.group(1))
        elif self.last_speed is not None:
            description += f" | last gen {self.last_speed:.1f} tok/s"
        emit({"type": "progress", "text": description})

    def set_postfix(self, *_args: Any, **_kwargs: Any) -> None:
        """Accept the tqdm postfix API without emitting a duplicate update."""


def worker_main(run_path: Path) -> int:
    """Load one model and execute coordinator tasks until stdin closes."""
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Worker must see exactly one CUDA GPU")

    config = load_config(run_path)
    dtype = str(config["model"].get("dtype", "auto")).lower()
    if dtype in {"bfloat16", "bf16"} and not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"{torch.cuda.get_device_name(0)} does not support {dtype}")

    samples = load_run_samples(run_path, config["dataset"])
    raw = {key: value for key, value in config.raw.items() if key != "_run_path"}
    raw["model"] = {**raw["model"], "device_map": {"": 0}}
    config = RunConfig.from_dict(run_path, raw)
    model, tokenizer = load_hf_model_and_tokenizer(config["model"])
    samples_per_item = int(config["generation"].get("num_samples_per_item", 1))
    emit({"type": "ready"})

    for line in sys.stdin:
        task = json.loads(line)
        if task["type"] == "stop":
            return 0
        sample_index = int(task["sample_index"])
        sample_iter = int(task["sample_iter"])
        sample = samples[sample_index]
        started = time.monotonic()
        try:
            label = (
                f"item {sample_index + 1}/{len(samples)} "
                f"{sample_id_from_sample(sample)} "
                f"iter {sample_iter + 1}/{samples_per_item}"
            )
            output = generate_task(
                run_path=run_path,
                config=config,
                model=model,
                tokenizer=tokenizer,
                sample=sample,
                sample_index=sample_index,
                sample_iter=sample_iter,
                progress=RemoteProgress(),
                progress_label=label,
            )
            emit(
                {
                    "type": "done",
                    "tokens": len(output.generated_token_ids),
                    "elapsed": time.monotonic() - started,
                }
            )
        except Exception as exc:
            emit(
                {
                    "type": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
    return 0


def parse_workers(nodes: list[str], devices: list[str]) -> list[tuple[str, int]]:
    """Expand one comma-separated GPU selection per SSH node."""
    if len(nodes) != len(devices):
        raise ValueError("--devices needs exactly one entry per --nodes host")
    workers = [
        (node, int(gpu))
        for node, selected in zip(nodes, devices)
        for gpu in selected.split(",")
        if gpu.strip()
    ]
    if not workers or len(workers) != len(set(workers)):
        raise ValueError("Worker GPU selections must be non-empty and unique")
    return workers


def pending_tasks(run_path: Path) -> tuple[list[dict[str, int]], int, int]:
    """Return unfinished tasks, total rollouts, and completed rollouts."""
    config = load_config(run_path)
    samples = load_run_samples(run_path, config["dataset"])
    generation_cfg = config["generation"]
    samples_per_item = int(generation_cfg.get("num_samples_per_item", 1))
    existing = load_generation_index(run_path)
    tasks = []
    complete = 0
    for sample_index, sample in enumerate(samples):
        for sample_iter in range(samples_per_item):
            key = generation_key_for(sample, sample_index, sample_iter, generation_cfg)
            if key in existing:
                complete += 1
            else:
                tasks.append({"sample_index": sample_index, "sample_iter": sample_iter})
    tasks.sort(key=lambda task: (task["sample_iter"], task["sample_index"]))
    return tasks, len(samples) * samples_per_item, complete


def receive(process: subprocess.Popen[str], log: TextIO) -> dict[str, Any]:
    """Read the next protocol message while logging ordinary worker stdout."""
    assert process.stdout is not None
    for line in process.stdout:
        if line.startswith(PREFIX):
            return json.loads(line[len(PREFIX) :])
        log.write(line)
        log.flush()
    raise RuntimeError(f"Worker exited with status {process.wait()}")


def worker_command(host: str, gpu: int, run_path: Path, root: Path) -> list[str]:
    """Build the SSH command for one selected physical GPU."""
    worker = f"{host}:{gpu}"
    remote = (
        f"cd {shlex.quote(root.as_posix())} && "
        f"export CUDA_VISIBLE_DEVICES={gpu} && "
        "exec bash scripts/run_with_hf_download_fix.sh "
        f".venv/bin/python -u scripts/generation/orchestrate.py "
        f"--run {shlex.quote(run_path.as_posix())} "
        f"--worker-id {shlex.quote(worker)}"
    )
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ServerAliveInterval=30",
        host,
        remote,
    ]


def run_worker(
    worker: tuple[str, int],
    run_path: Path,
    root: Path,
    tasks: Queue[dict[str, int]],
    total_bar: Any,
    worker_bar: Any,
    lock: RLock,
    stop: Event,
    processes: list[subprocess.Popen[str]],
) -> None:
    """Load one remote model and consume tasks until the shared queue is empty."""
    host, gpu = worker
    name = f"{host}:{gpu}"
    log_path = run_path / "generation" / "orchestrator_logs" / f"{host}_{gpu}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            worker_command(host, gpu, run_path, root),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            bufsize=1,
        )
        with lock:
            processes.append(process)
        try:
            if receive(process, log).get("type") != "ready":
                raise RuntimeError(f"{name} failed during model loading")
            worker_bar.set_description_str(f"{name:<20} ready", refresh=True)
            while not stop.is_set():
                try:
                    task = tasks.get_nowait()
                except Empty:
                    break
                assert process.stdin is not None
                process.stdin.write(json.dumps({"type": "task", **task}) + "\n")
                process.stdin.flush()
                while True:
                    event = receive(process, log)
                    if event["type"] == "progress":
                        worker_bar.set_description_str(
                            f"{name:<20} {event['text']}", refresh=True
                        )
                    elif event["type"] == "done":
                        speed = event["tokens"] / max(event["elapsed"], 1e-9)
                        with lock:
                            total_bar.update(1)
                            worker_bar.set_description_str(
                                f"{name:<20} done | {speed:.1f} tok/s overall",
                                refresh=True,
                            )
                        break
                    else:
                        raise RuntimeError(
                            f"{name}: {event['error']}\n{event['traceback']}"
                        )
            if process.poll() is None:
                assert process.stdin is not None
                process.stdin.write('{"type":"stop"}\n')
                process.stdin.flush()
                process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
            with lock:
                processes.remove(process)


def orchestrate(
    workers: list[tuple[str, int]],
    run_path: Path,
    remote_root: Path,
) -> None:
    """Run the dynamic queue with one fixed tqdm line per selected GPU."""
    pending, total, complete = pending_tasks(run_path)
    if not pending:
        print(f"All {total} rollouts are already complete.")
        return

    tasks: Queue[dict[str, int]] = Queue()
    for task in pending:
        tasks.put(task)
    total_bar = tqdm(
        total=total,
        initial=complete,
        desc="total",
        unit="rollout",
        position=0,
        dynamic_ncols=True,
    )
    bars = [
        tqdm(
            total=0,
            desc=f"{f'{host}:{gpu}':<20} starting",
            position=index + 1,
            bar_format="{desc}",
            leave=True,
        )
        for index, (host, gpu) in enumerate(workers)
    ]
    lock = RLock()
    stop = Event()
    processes: list[subprocess.Popen[str]] = []
    pool = ThreadPoolExecutor(max_workers=len(workers))
    futures = [
        pool.submit(
            run_worker,
            worker,
            run_path,
            remote_root,
            tasks,
            total_bar,
            bar,
            lock,
            stop,
            processes,
        )
        for worker, bar in zip(workers, bars)
    ]
    try:
        for future in as_completed(futures):
            future.result()
    except BaseException:
        stop.set()
        with lock:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        for bar in bars:
            bar.close()
        total_bar.close()


def main() -> int:
    """Run as the lamgate coordinator or as its hidden remote worker mode."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", nargs="+")
    parser.add_argument("--devices", nargs="+")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--worker-id", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker_id:
        return worker_main(args.run)
    if not args.nodes or not args.devices:
        parser.error("--nodes and --devices are required")

    root = Path.cwd().resolve()
    run_path = args.run
    if run_path.is_absolute():
        run_path = run_path.relative_to(root)
    orchestrate(parse_workers(args.nodes, args.devices), run_path, root)
    return 0
