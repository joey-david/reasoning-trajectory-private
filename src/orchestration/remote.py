"""Distribute resumable job tasks to one persistent process per selected GPU."""

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

from src.orchestration.jobs import load_job
from src.orchestration.jobs.contract import Task
from src.runtime.config import load_config


PREFIX = "@@ORCHESTRATOR@@"


def emit(message: dict[str, Any]) -> None:
    """Write one worker control message."""
    print(PREFIX + json.dumps(message, ensure_ascii=False), flush=True)


class WorkerProgress:
    """Forward optional task progress while retaining reported token speed."""

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


def worker_main(run_path: Path, job_name: str) -> int:
    """Set up one job worker and execute coordinator tasks until stdin closes."""
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Worker must see exactly one CUDA GPU")

    config = load_config(run_path)
    dtype = str(config.get("model", {}).get("dtype", "auto")).lower()
    if dtype in {"bfloat16", "bf16"} and not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"{torch.cuda.get_device_name(0)} does not support {dtype}")

    worker = load_job(job_name).setup_worker(run_path)
    emit({"type": "ready"})

    for line in sys.stdin:
        task = json.loads(line)
        if task["type"] == "stop":
            return 0
        started = time.monotonic()
        try:
            result = worker.run_task(task, WorkerProgress())
            emit(
                {
                    "type": "done",
                    "units": result.units,
                    "unit": result.unit,
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


def receive(process: subprocess.Popen[str], log: TextIO) -> dict[str, Any]:
    """Read the next protocol message while logging ordinary worker stdout."""
    assert process.stdout is not None
    for line in process.stdout:
        if line.startswith(PREFIX):
            return json.loads(line[len(PREFIX) :])
        log.write(line)
        log.flush()
    raise RuntimeError(f"Worker exited with status {process.wait()}")


def worker_command(
    host: str,
    gpu: int,
    run_path: Path,
    root: Path,
    job_name: str,
) -> list[str]:
    """Build a local or SSH command for one selected physical GPU."""
    worker = f"{host}:{gpu}"
    command = (
        f"cd {shlex.quote(root.as_posix())} && "
        f"export CUDA_VISIBLE_DEVICES={gpu} && "
        "exec bash scripts/run_with_hf_download_fix.sh "
        f".venv/bin/python -u scripts/orchestrate.py "
        f"--run {shlex.quote(run_path.as_posix())} "
        f"--job {shlex.quote(job_name)} "
        f"--worker-id {shlex.quote(worker)}"
    )
    if host == "local":
        return ["bash", "-lc", command]
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ServerAliveInterval=30",
        host,
        command,
    ]


def run_worker(
    worker: tuple[str, int],
    job_name: str,
    run_path: Path,
    root: Path,
    tasks: Queue[Task],
    total_bar: Any,
    worker_bar: Any,
    lock: RLock,
    stop: Event,
    processes: list[subprocess.Popen[str]],
) -> None:
    """Load one persistent worker and consume tasks until the queue is empty."""
    host, gpu = worker
    name = f"{host}:{gpu}"
    log_path = load_job(job_name).log_path(run_path, host, gpu)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            worker_command(host, gpu, run_path, root, job_name),
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
                        speed = event["units"] / max(event["elapsed"], 1e-9)
                        with lock:
                            total_bar.update(1)
                            worker_bar.set_description_str(
                                f"{name:<20} done | {speed:.1f} "
                                f"{event['unit']}/s overall",
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
    job_name: str = "generation",
) -> None:
    """Run the dynamic queue with one fixed tqdm line per selected GPU."""
    pending, total, complete = load_job(job_name).pending_tasks(run_path)
    if not pending:
        print(f"All {total} tasks are already complete.")
        return

    tasks: Queue[Task] = Queue()
    for task in pending:
        tasks.put(task)
    total_bar = tqdm(
        total=total,
        initial=complete,
        desc="total",
        unit="task",
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
            job_name,
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
    parser.add_argument("--job", default="generation")
    parser.add_argument("--worker-id", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker_id:
        return worker_main(args.run, args.job)
    if not args.nodes or not args.devices:
        parser.error("--nodes and --devices are required")

    root = Path.cwd().resolve()
    run_path = args.run
    if run_path.is_absolute():
        run_path = run_path.relative_to(root)
    orchestrate(
        parse_workers(args.nodes, args.devices),
        run_path,
        root,
        job_name=args.job,
    )
    return 0
