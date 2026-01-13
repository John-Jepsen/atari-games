from __future__ import annotations

import numpy as np


def format_obs(obs: np.ndarray, observation_type: str | None) -> np.ndarray:
    if observation_type != "pixels":
        return obs.astype(np.float32)
    if obs.ndim == 3 and obs.shape[0] != 4 and obs.shape[-1] == 4:
        obs = np.transpose(obs, (2, 0, 1))
    return obs.astype(np.uint8)
