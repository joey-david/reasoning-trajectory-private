from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from src.runtime.config import load_config
from src.datasets.loaders import load_run_samples


def generate_runs(run_paths: list[Path]) -> None:
    for index, run_path in enumerate(run_paths, start=1):
        print(f"[{index}/{len(run_paths)}] generating {run_path}", flush=True)
        generate_one_run(run_path)


def generate_one_run(run_path: Path) -> None:
    config = load_config(run_path)
    samples = load_run_samples(run_path, config["dataset"])
    devices = replica_devices(config["model"].get("device_map"))
    if len(devices) > 1:
        generate_parallel(run_path, config.raw, samples, devices)
        return

    from src.models.generation_pipeline import generate_run

    generate_run(run_path, config, samples)


def replica_devices(device_map: Any) -> list[int]:
    if not isinstance(device_map, dict):
        return []
    placement = device_map.get("")
    if isinstance(placement, list):
        devices = [int(device) for device in placement]
    elif isinstance(placement, str) and "," in placement:
        devices = [int(device.strip()) for device in placement.split(",")]
    else:
        return []
    if len(devices) != len(set(devices)):
        raise ValueError(f"Duplicate replica devices: {devices}")
    return devices


def generate_parallel(
    run_path: Path,
    config: dict[str, Any],
    samples: list[dict[str, Any]],
    devices: list[int],
) -> None:
    ranges = contiguous_ranges(len(samples), len(devices))
    raw_config = {key: value for key, value in config.items() if key != "_run_path"}
    with ProcessPoolExecutor(
        max_workers=len(devices),
        mp_context=get_context("spawn"),
    ) as pool:
        futures = [
            pool.submit(
                generate_shard,
                run_path,
                raw_config,
                samples[start:stop],
                start,
                device,
            )
            for device, (start, stop) in zip(devices, ranges)
            if start < stop
        ]
        for future in futures:
            future.result()


def generate_shard(
    run_path: Path,
    config: dict[str, Any],
    samples: list[dict[str, Any]],
    sample_index_offset: int,
    device: int,
) -> None:
    from src.runtime.config import RunConfig
    from src.models.generation_pipeline import generate_run

    model = {**config["model"], "device_map": {"": device}}
    worker_config = RunConfig.from_dict(run_path, {**config, "model": model})
    print(
        f"[gpu {device}] generating {len(samples)} items "
        f"from index {sample_index_offset}",
        flush=True,
    )
    generate_run(
        run_path,
        worker_config,
        samples,
        sample_index_offset=sample_index_offset,
    )


def contiguous_ranges(total: int, parts: int) -> list[tuple[int, int]]:
    return [
        (total * rank // parts, total * (rank + 1) // parts)
        for rank in range(parts)
    ]
