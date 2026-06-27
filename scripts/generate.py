#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate outputs for one or more run folders."
    )
    parser.add_argument(
        "run_paths", nargs="+", help="Run folder(s), executed sequentially."
    )
    args = parser.parse_args()

    for i, run_path_arg in enumerate(args.run_paths, start=1):
        run_path = Path(run_path_arg)
        print(f"[{i}/{len(args.run_paths)}] generating {run_path}", flush=True)
        generate_one_run(run_path)
    return 0


def generate_one_run(run_path: Path) -> None:
    from src.config import load_config
    from src.datasets.loaders import load_run_samples

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
    from src.config import RunConfig
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


if __name__ == "__main__":
    raise SystemExit(main())
