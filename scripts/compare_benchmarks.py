#!/usr/bin/env python3
from __future__ import annotations

"""Run CPU vs MPS (or CUDA) benchmarks in sequence and summarize results.

This uses the project's benchmark script so results reflect the full DQN loop
(experience replay, target network, frame-stacked inputs for Atari).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _run_one(
    config: str,
    device: str,
    frames: int,
    mode: str,
    tag: str,
    python: str,
    workdir: str,
) -> dict[str, Any]:
    env = os.environ.copy()
    repo_root = Path(workdir)
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = src_path + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    cmd = [
        python,
        "scripts/benchmark_training.py",
        "--config",
        config,
        "--device",
        device,
        "--frames",
        str(frames),
        "--mode",
        mode,
        "--tag",
        tag,
    ]
    result = subprocess.run(cmd, cwd=workdir, env=env, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Benchmark failed for {device} (exit {result.returncode}).")

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse benchmark output for {device}.") from exc


def _format_fps(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _summarize(results: list[dict[str, Any]]) -> None:
    print("\n=== Benchmark summary ===")
    for payload in results:
        device = payload.get("device_used")
        env_fps = _format_fps(payload.get("env_fps"))
        train_fps = _format_fps(payload.get("train_fps"))
        print(f"{device}: env_fps={env_fps} train_fps={train_fps}")

    train_sorted = [r for r in results if r.get("train_fps") is not None]
    if train_sorted:
        best = max(train_sorted, key=lambda r: r.get("train_fps") or 0.0)
        print(f"\nBest training throughput: {best.get('device_used')} ({_format_fps(best.get('train_fps'))} fps)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare DQN benchmark throughput across devices.")
    parser.add_argument("--config", required=True, help="Path to a JSON config file.")
    parser.add_argument("--frames", type=int, default=20000, help="Number of frames per run.")
    parser.add_argument(
        "--devices",
        default="mps,cpu",
        help="Comma-separated device list (e.g., mps,cpu or cuda,cpu).",
    )
    parser.add_argument(
        "--mode",
        default="both",
        choices=["env", "train", "both"],
        help="Benchmark env-only, train-only, or both.",
    )
    parser.add_argument("--tag", default="", help="Optional tag for output grouping.")
    args = parser.parse_args()

    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    if not devices:
        raise SystemExit("No devices specified.")

    run_tag = args.tag.strip() or time.strftime("%Y%m%d_%H%M%S")
    python = sys.executable
    workdir = os.getcwd()

    results = []
    for device in devices:
        print(f"\n=== Running {device} ===")
        payload = _run_one(
            config=args.config,
            device=device,
            frames=args.frames,
            mode=args.mode,
            tag=run_tag,
            python=python,
            workdir=workdir,
        )
        results.append(payload)

    _summarize(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
