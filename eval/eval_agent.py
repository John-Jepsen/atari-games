#!/usr/bin/env python3
from __future__ import annotations

import argparse

from atari_games.config import load_config, require_keys
from atari_games.envs import make_atari, make_cartpole
from atari_games.preprocess import format_obs
from atari_games.trainer import _build_agent, evaluate_agent, load_checkpoint
from atari_games.utils import get_device, seed_everything, to_numpy


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN agent.")
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
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    require_keys(cfg, ["env_id", "seed", "dqn", "training"], "root")
    seed_everything(int(cfg.get("seed", 42)))

    device = get_device(cfg.get("device"))
    if cfg.get("observation_type") == "pixels":
        env = make_atari(cfg["env_id"], seed=int(cfg.get("seed", 42)), preprocess=cfg["preprocess"])
    else:
        env = make_cartpole(cfg["env_id"], seed=int(cfg.get("seed", 42)))

    obs, _ = env.reset()
    obs = format_obs(to_numpy(obs), cfg.get("observation_type"))
    obs_shape = obs.shape
    num_actions = env.action_space.n

    agent = _build_agent(cfg, obs_shape, num_actions, device)
    load_checkpoint(args.checkpoint, agent)

    score = evaluate_agent(
        agent,
        env,
        args.episodes,
        epsilon=float(cfg["training"].get("eval_epsilon", 0.05)),
        observation_type=cfg.get("observation_type"),
    )
    print(f"Average score over {args.episodes} episodes: {score:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
