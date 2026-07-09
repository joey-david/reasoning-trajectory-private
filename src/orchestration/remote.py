"""Distribute resumable job tasks to one persistent process per selected GPU."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
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
GpuGroup = tuple[int, ...]
Worker = tuple[str, GpuGroup]


def emit(message: dict[str, Any]) -> None:
    """Write one worker control message.

    Args:
        message: JSON-serializable worker protocol message.

    Returns:
        None.
    """
    print(PREFIX + json.dumps(message, ensure_ascii=False), flush=True)


class WorkerProgress:
    """Forward optional task progress while retaining reported token speed."""

    def __init__(self) -> None:
        """Initialize the helper state.

        Args:
            None.

        Returns:
            None.
        """
        self.last_speed: float | None = None

    def set_description(self, description: str, **_kwargs: Any) -> None:
        """Forward a tqdm-compatible progress description.

        Args:
            description: Text to display for the current progress state.
            _kwargs: Ignored keyword arguments accepted for tqdm compatibility.

        Returns:
            None.
        """
        match = re.search(r"(\d+(?:\.\d+)?) tok/s", description)
        if match:
            self.last_speed = float(match.group(1))
        elif self.last_speed is not None:
            description += f" | last gen {self.last_speed:.1f} tok/s"
        emit({"type": "progress", "text": description})

    def set_postfix(self, *_args: Any, **_kwargs: Any) -> None:
        """Accept the tqdm postfix API without emitting a duplicate update.

        Args:
            _args: Ignored positional arguments accepted for tqdm compatibility.
            _kwargs: Ignored keyword arguments accepted for tqdm compatibility.

        Returns:
            None.
        """


def worker_main(run_path: Path, job_name: str) -> int:
    """Set up one job worker and execute coordinator tasks until stdin closes.

    Args:
        run_path: Run directory containing the configuration and artifacts.
        job_name: Registered orchestration job name.

    Returns:
        The computed index, count, or status code.
    """
    import torch

    config = load_config(run_path)
    visible_group_size = int(os.environ.get("ORCHESTRATOR_GPU_COUNT", "1"))
    expected_gpus = int(
        config.get("model", {}).get("required_gpus", visible_group_size)
    )
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != expected_gpus
        or visible_group_size != expected_gpus
    ):
        raise RuntimeError(
            f"Worker must see exactly {expected_gpus} CUDA GPU(s), "
            f"launcher declared {visible_group_size} and PyTorch found "
            f"{torch.cuda.device_count()}"
        )

    dtype = str(config.get("model", {}).get("dtype", "auto")).lower()
    if dtype in {"bfloat16", "bf16"} and not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"{torch.cuda.get_device_name(0)} does not support {dtype}")

    emit(
        {
            "type": "loading",
            "text": f"loading {config.get('model', {}).get('name', 'worker')}",
        }
    )
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


def parse_workers(nodes: list[str], devices: list[str]) -> list[Worker]:
    """Parse independent or grouped GPU selections for each SSH node.

    Args:
        nodes: Remote host names.
        devices: Per-host selections; commas split workers and plus signs group GPUs.

    Returns:
        The resulting ordered records or values.
    """
    if len(nodes) != len(devices):
        raise ValueError("--devices needs exactly one entry per --nodes host")
    workers = [
        (node, tuple(int(gpu) for gpu in group.split("+")))
        for node, selected in zip(nodes, devices)
        for group in selected.split(",")
        if group.strip()
    ]
    selected_devices = [(node, gpu) for node, group in workers for gpu in group]
    if (
        not workers
        or any(not group or len(group) != len(set(group)) for _, group in workers)
        or len(selected_devices) != len(set(selected_devices))
    ):
        raise ValueError("Worker GPU selections must be non-empty and disjoint")
    return workers


def receive(process: subprocess.Popen[str], log: TextIO) -> dict[str, Any]:
    """Read the next protocol message while logging ordinary worker stdout.

    Args:
        process: Worker subprocess from which to receive messages.
        log: Open worker log stream.

    Returns:
        The resulting keyed records or metrics.
    """
    assert process.stdout is not None
    for line in process.stdout:
        if line.startswith(PREFIX):
            return json.loads(line[len(PREFIX) :])
        log.write(line)
        log.flush()
    raise RuntimeError(f"Worker exited with status {process.wait()}")


def worker_command(
    host: str,
    gpu: int | GpuGroup,
    run_path: Path,
    root: Path,
    job_name: str,
) -> list[str]:
    """Build a local or SSH command for one selected physical GPU.

    Args:
        host: Remote worker host name.
        gpu: One GPU index or a grouped set used by one model worker.
        run_path: Run directory containing the configuration and artifacts.
        root: Repository root on the remote hosts.
        job_name: Registered orchestration job name.

    Returns:
        The resulting ordered records or values.
    """
    group = (gpu,) if isinstance(gpu, int) else gpu
    devices = ",".join(str(device) for device in group)
    worker = f"{host}:{'+'.join(str(device) for device in group)}"
    token_exports = " ".join(
        f"export {name}={shlex.quote(value)} &&"
        for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
        if (value := os.environ.get(name))
    )
    command = (
        f"cd {shlex.quote(root.as_posix())} && "
        f"{token_exports} "
        f"export CUDA_VISIBLE_DEVICES={devices} && "
        f"export ORCHESTRATOR_GPU_COUNT={len(group)} && "
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
    worker: Worker,
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
    """Load one persistent worker and consume tasks until the queue is empty.

    Args:
        worker: Remote host and grouped GPU worker specification.
        job_name: Registered orchestration job name.
        run_path: Run directory containing the configuration and artifacts.
        root: Repository root on the remote hosts.
        tasks: Serialized tasks assigned to the worker.
        total_bar: Shared overall progress display.
        worker_bar: Per-worker progress display.
        lock: Lock protecting shared progress and process state.
        stop: Event requesting all workers to stop.
        processes: Mutable registry of active worker processes.

    Returns:
        None.
    """
    host, gpu_group = worker
    gpu_label = "+".join(str(gpu) for gpu in gpu_group)
    name = f"{host}:{gpu_label}"
    log_path = load_job(job_name).log_path(run_path, host, gpu_label)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            worker_command(host, gpu_group, run_path, root, job_name),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            bufsize=1,
        )
        with lock:
            processes.append(process)
        try:
            try:
                # Loading messages can be arbitrarily numerous; only `ready`
                # transitions the protocol into task dispatch.
                while True:
                    startup = receive(process, log)
                    if startup.get("type") == "loading":
                        worker_bar.set_description_str(
                            f"{name:<20} {startup['text']}",
                            refresh=True,
                        )
                        continue
                    if startup.get("type") == "ready":
                        break
                    raise RuntimeError(f"{name} failed during model loading")
            except Exception as error:
                log.write(f"\ncoordinator: startup failed: {error}\n")
                log.flush()
                worker_bar.set_description_str(
                    f"{name:<20} startup failed | see {log_path}",
                    refresh=True,
                )
                return
            worker_bar.set_description_str(f"{name:<20} ready", refresh=True)
            while not stop.is_set():
                try:
                    task = tasks.get_nowait()
                except Empty:
                    break
                assert process.stdin is not None
                process.stdin.write(json.dumps({"type": "task", **task}) + "\n")
                process.stdin.flush()
                # One task may emit many progress events but exactly one terminal
                # done/error event before the next task can be sent.
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
    workers: list[Worker],
    run_path: Path,
    remote_root: Path,
    job_name: str = "generation",
) -> None:
    """Run the dynamic queue with one fixed tqdm line per selected GPU.

    Args:
        workers: Remote host/GPU worker specifications.
        run_path: Run directory containing the configuration and artifacts.
        remote_root: Repository root on the remote hosts.
        job_name: Registered orchestration job name.

    Returns:
        None.
    """
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
            desc=(
                f"{host}:{'+'.join(str(value) for value in gpu)}".ljust(20)
                + " starting"
            ),
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
        remaining = tasks.qsize()
        if remaining:
            raise RuntimeError(
                f"{remaining}/{len(pending)} pending tasks remain because no "
                f"worker could run them; inspect {job_name} orchestrator logs"
            )
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
    """Run as the lamgate coordinator or as its hidden remote worker mode.

    Args:
        None.

    Returns:
        The computed index, count, or status code.
    """
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
