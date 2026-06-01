#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import termios
import tty
from datetime import datetime, timezone
from pathlib import Path

from reasoning_trajectory.core.storage import load_trajectories, save_jsonl


def tail_words(text: str, n: int) -> str:
    words = text.split()
    return " ".join(words[-n:])


def read_key() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1).lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main() -> int:
    parser = argparse.ArgumentParser(description="Manually label trajectory correctness from generation tails.")
    parser.add_argument("run", help="Run directory or trajectory JSONL.")
    parser.add_argument("--tail-words", type=int, default=100)
    parser.add_argument("--write", action="store_true", help="Update trajectories.jsonl in place after making a backup.")
    parser.add_argument("--labels-out", default=None, help="Optional JSONL file for labels only.")
    args = parser.parse_args()

    run = Path(args.run)
    jsonl = run / "trajectories.jsonl" if run.is_dir() else run
    trajectories = load_trajectories(run)
    labels = []
    for i, traj in enumerate(trajectories, 1):
        if traj.final_correct is not None:
            current = f"currently={traj.final_correct}"
        else:
            current = "currently=unlabeled"
        print("\n" + "=" * 88)
        print(f"{i}/{len(trajectories)} {traj.trajectory_id} seed={traj.seed} {current}")
        print(f"problem: {traj.prompt}")
        print("-" * 88)
        print(tail_words(traj.final_text, args.tail_words))
        print("-" * 88)
        print("y=true  n=false  u=unknown  s=skip  q=quit")
        key = read_key()
        print(key)
        if key == "q":
            break
        if key == "s":
            continue
        if key not in {"y", "n", "u"}:
            print("ignored")
            continue
        value = True if key == "y" else False if key == "n" else None
        traj.final_correct = value
        traj.metadata["manual_label"] = {
            "correct": value,
            "labeled_at": datetime.now(timezone.utc).isoformat(),
            "tail_words": args.tail_words,
        }
        labels.append({"trajectory_id": traj.trajectory_id, "seed": traj.seed, "manual_correct": value})

    out = Path(args.labels_out) if args.labels_out else (jsonl.parent / "manual_labels.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in labels:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote labels: {out}")
    if args.write and labels:
        backup = jsonl.with_suffix(jsonl.suffix + f".bak-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
        shutil.copy2(jsonl, backup)
        save_jsonl(trajectories, jsonl)
        print(f"updated trajectories: {jsonl}")
        print(f"backup: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
