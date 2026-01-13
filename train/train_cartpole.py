#!/usr/bin/env python3
from __future__ import annotations

import argparse

from atari_games.config import load_config, require_keys, summarize_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a CartPole agent (scaffold).")
    parser.add_argument(
        "--config",
        default="configs/cartpole.json",
        help="Path to a JSON config file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    require_keys(cfg, ["env_id", "seed", "dqn", "training"], "root")

    print("Loaded config:")
    print(summarize_config(cfg))
    print("\nScaffold only: core DQN components will be implemented in Step 3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
