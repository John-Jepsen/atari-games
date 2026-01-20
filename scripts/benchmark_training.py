#!/usr/bin/env python3
from __future__ import annotations

"""
Benchmark DQN throughput with the same training loop used in this project.

This aligns with the referenced DQN papers (experience replay, target network,
frame-stacked inputs for Atari) so speed comparisons reflect the real workload.
"""

import argparse
import json
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atari_games.config import load_config, require_keys, summarize_config
from atari_games.envs import make_atari, make_cartpole
from atari_games.preprocess import format_obs
from atari_games.trainer import train_from_config
from atari_games.utils import get_device, seed_everything, to_numpy


@dataclass
class BenchmarkResult:
    device_request: str
    device_used: str
    env_fps: float | None
    train_fps: float | None
    env_seconds: float | None
    train_seconds: float | None
    frames: int
    output_dir: str | None
    checkpoint_dir: str | None


def _make_env(cfg: dict[str, Any], seed: int):
    if cfg.get("observation_type") == "pixels":
        return make_atari(cfg["env_id"], seed=seed, preprocess=cfg["preprocess"])
    return make_cartpole(cfg["env_id"], seed=seed)


def _benchmark_env(cfg: dict[str, Any], frames: int) -> tuple[float, float]:
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    env = _make_env(cfg, seed=seed)
    obs, _ = env.reset()
    _ = format_obs(to_numpy(obs), cfg.get("observation_type"))

    start = time.perf_counter()
    for _ in range(frames):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    elapsed = time.perf_counter() - start
    fps = frames / elapsed if elapsed > 0 else 0.0
    return fps, elapsed


def _benchmark_train(
    cfg: dict[str, Any],
    frames: int,
    device: str,
    output_dir: Path,
    checkpoint_dir: Path,
) -> tuple[float, float]:
    cfg_copy = deepcopy(cfg)
    cfg_copy["device"] = device
    cfg_copy.setdefault("training", {})
    cfg_copy["training"]["total_frames"] = frames
    cfg_copy["training"]["eval_every_frames"] = 0
    cfg_copy["training"]["eval_episodes"] = 0
    cfg_copy["training"]["checkpoint_every_frames"] = 0

    start = time.perf_counter()
    _ = train_from_config(
        cfg_copy,
        output_dir=str(output_dir),
        checkpoint_dir=str(checkpoint_dir),
    )
    elapsed = time.perf_counter() - start
    fps = frames / elapsed if elapsed > 0 else 0.0
    return fps, elapsed


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark DQN throughput (env + training).")
    parser.add_argument("--config", required=True, help="Path to a JSON config file.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Device request (auto chooses best available).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=20000,
        help="Number of frames to benchmark for env and training.",
    )
    parser.add_argument(
        "--mode",
        default="both",
        choices=["env", "train", "both"],
        help="Benchmark env-only, train-only, or both.",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional run tag for output folders (default: timestamp).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    require_keys(cfg, ["env_id", "seed", "dqn", "training"], "root")

    device_request = args.device.lower()
    device_for_cfg = None if device_request == "auto" else device_request
    device_used = str(get_device(device_for_cfg))

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_tag = args.tag.strip() or timestamp
    env_name = cfg["env_id"].replace("/", "_")

    reports_root = Path("reports") / "benchmarks" / run_tag / device_used
    models_root = Path("models") / "benchmarks" / run_tag / device_used
    summary_path = Path("reports") / "benchmarks" / run_tag / f"benchmark_{env_name}_{device_used}.json"

    env_fps = env_seconds = None
    train_fps = train_seconds = None

    if args.mode in {"env", "both"}:
        env_fps, env_seconds = _benchmark_env(cfg, args.frames)

    if args.mode in {"train", "both"}:
        train_fps, train_seconds = _benchmark_train(
            cfg,
            args.frames,
            device=device_used,
            output_dir=reports_root,
            checkpoint_dir=models_root,
        )

    result = BenchmarkResult(
        device_request=device_request,
        device_used=device_used,
        env_fps=env_fps,
        train_fps=train_fps,
        env_seconds=env_seconds,
        train_seconds=train_seconds,
        frames=args.frames,
        output_dir=str(reports_root) if args.mode in {"train", "both"} else None,
        checkpoint_dir=str(models_root) if args.mode in {"train", "both"} else None,
    )

    payload = {
        "timestamp": time.time(),
        "run_tag": run_tag,
        "config_path": args.config,
        "config_summary": summarize_config(cfg),
        "env_id": cfg["env_id"],
        "device_request": result.device_request,
        "device_used": result.device_used,
        "frames": result.frames,
        "env_fps": result.env_fps,
        "train_fps": result.train_fps,
        "env_seconds": result.env_seconds,
        "train_seconds": result.train_seconds,
        "reports_dir": result.output_dir,
        "checkpoint_dir": result.checkpoint_dir,
        "mode": args.mode,
    }

    _write_summary(summary_path, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
