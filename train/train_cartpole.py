#!/usr/bin/env python3
from __future__ import annotations

import argparse

from atari_games.config import load_config, require_keys
from atari_games.trainer import dump_run_summary, train_from_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a CartPole agent (DQN).")
    parser.add_argument(
        "--config",
        default="configs/cartpole.json",
        help="Path to a JSON config file.",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Log params, metrics, and artifacts to MLflow.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    require_keys(cfg, ["env_id", "seed", "dqn", "training"], "root")
    if args.mlflow:
        cfg["training"]["mlflow"] = True

    result = train_from_config(cfg)
    dump_run_summary(cfg, result)
    print(f"Saved metrics to {result.metrics_path}")
    print(f"Saved checkpoint to {result.checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
