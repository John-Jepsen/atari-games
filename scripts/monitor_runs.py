#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


@dataclass
class MetricSnapshot:
    env_name: str
    episodes: int
    last_reward: float
    last_epsilon: float
    last_loss: float
    avg_reward: float
    path: Path
    updated_seconds_ago: int


def _run_ps() -> List[str]:
    try:
        output = subprocess.check_output(["ps", "-ax", "-o", "pid=,command="]).decode()
    except Exception:
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def find_training_pids() -> List[Tuple[str, int]]:
    pids: List[Tuple[str, int]] = []
    for line in _run_ps():
        if "train/train_cartpole.py" in line or "train/train_atari.py" in line:
            try:
                pid_str, cmd = line.split(" ", 1)
                pids.append((cmd, int(pid_str)))
            except ValueError:
                continue
    return pids


def _tail_lines(path: Path, count: int) -> List[str]:
    if count <= 0:
        return []
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = 4096
            data = b""
            while size > 0 and data.count(b"\n") <= count:
                read_size = min(block, size)
                f.seek(size - read_size)
                data = f.read(read_size) + data
                size -= read_size
        lines = data.splitlines()[-count:]
        return [line.decode() for line in lines]
    except Exception:
        return []


def _parse_metrics(path: Path, tail: int) -> Optional[MetricSnapshot]:
    lines = _tail_lines(path, tail + 1)
    if len(lines) <= 1:
        return None
    records = []
    for line in lines:
        if line.startswith("frame,"):
            continue
        parts = line.split(",")
        if len(parts) < 5:
            continue
        try:
            frame = int(parts[0])
            episode = int(parts[1])
            reward = float(parts[2])
            epsilon = float(parts[3])
            loss = float(parts[4])
            records.append((frame, episode, reward, epsilon, loss))
        except ValueError:
            continue
    if not records:
        return None

    last = records[-1]
    avg_reward = sum(r[2] for r in records) / float(len(records))
    env_name = path.name.replace("metrics_", "").replace(".csv", "")
    updated_seconds_ago = int(time.time() - path.stat().st_mtime)
    return MetricSnapshot(
        env_name=env_name,
        episodes=last[1],
        last_reward=last[2],
        last_epsilon=last[3],
        last_loss=last[4],
        avg_reward=avg_reward,
        path=path,
        updated_seconds_ago=updated_seconds_ago,
    )


def collect_metrics(reports_dir: Path, tail: int) -> List[MetricSnapshot]:
    snapshots = []
    for path in sorted(reports_dir.glob("metrics_*.csv")):
        snapshot = _parse_metrics(path, tail)
        if snapshot:
            snapshots.append(snapshot)
    return snapshots


def print_status(pids: List[Tuple[str, int]], snapshots: List[MetricSnapshot]) -> None:
    print("\n=== Training Processes ===")
    if not pids:
        print("No training processes found.")
    else:
        for cmd, pid in pids:
            print(f"PID {pid}: {cmd}")

    print("\n=== Metrics (last episodes) ===")
    if not snapshots:
        print("No metrics found.")
        return
    for snap in snapshots:
        print(
            f"{snap.env_name}: episodes={snap.episodes} "
            f"last_reward={snap.last_reward:.2f} avg_reward={snap.avg_reward:.2f} "
            f"epsilon={snap.last_epsilon:.3f} loss={snap.last_loss:.4f} "
            f"updated={snap.updated_seconds_ago}s ago"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor long-running DQN training runs.")
    parser.add_argument("--reports", default="reports", help="Reports directory path.")
    parser.add_argument("--tail", type=int, default=20, help="Episodes to average over.")
    parser.add_argument("--interval", type=int, default=30, help="Refresh interval in seconds.")
    parser.add_argument("--once", action="store_true", help="Print once and exit.")
    args = parser.parse_args()

    reports_dir = Path(args.reports)

    while True:
        pids = find_training_pids()
        snapshots = collect_metrics(reports_dir, args.tail)
        print_status(pids, snapshots)
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
