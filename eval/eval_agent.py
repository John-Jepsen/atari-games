#!/usr/bin/env python3
from __future__ import annotations

import argparse

from atari_games.config import load_config, require_keys, summarize_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained agent (scaffold).")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a JSON config file.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a saved model checkpoint.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    require_keys(cfg, ["env_id", "seed", "dqn", "training"], "root")

    print("Loaded config:")
    print(summarize_config(cfg))
    print(f"Checkpoint: {args.checkpoint}")
    print("\nScaffold only: evaluation pipeline will be implemented in Step 3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
