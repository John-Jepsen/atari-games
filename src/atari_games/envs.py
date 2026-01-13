from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, RecordEpisodeStatistics


def make_cartpole(env_id: str, seed: int) -> gym.Env:
    env = gym.make(env_id)
    env = RecordEpisodeStatistics(env)
    env.reset(seed=seed)
    return env


def make_atari(env_id: str, seed: int, preprocess: dict[str, Any]) -> gym.Env:
    env = gym.make(env_id)
    env = AtariPreprocessing(
        env,
        noop_max=preprocess.get("no_op_max", 30),
        frame_skip=preprocess.get("frame_skip", 4),
        screen_size=preprocess.get("resize_width", 84),
        terminal_on_life_loss=preprocess.get("terminal_on_life_loss", True),
        grayscale_obs=preprocess.get("grayscale", True),
        scale_obs=False,
    )
    env = FrameStackObservation(env, stack_size=preprocess.get("frame_stack", 4))
    env = RecordEpisodeStatistics(env)
    env.reset(seed=seed)
    return env
