from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


@dataclass
class EpsilonSchedule:
    start: float
    end: float
    decay_frames: int

    def value(self, frame_idx: int) -> float:
        if frame_idx >= self.decay_frames:
            return self.end
        slope = (self.end - self.start) / float(self.decay_frames)
        return self.start + slope * frame_idx


@dataclass
class LinearSchedule:
    start: float
    end: float
    duration_frames: int

    def value(self, frame_idx: int) -> float:
        if frame_idx >= self.duration_frames:
            return self.end
        slope = (self.end - self.start) / float(self.duration_frames)
        return self.start + slope * frame_idx


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(requested: str | None = None) -> torch.device:
    if requested:
        requested = requested.lower()
        if requested == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            raise RuntimeError("CUDA requested but not available.")
        if requested == "mps":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            raise RuntimeError("MPS requested but not available.")
        if requested == "cpu":
            return torch.device("cpu")
        if requested != "auto":
            raise RuntimeError(f"Unknown device request: {requested}")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_numpy(obs) -> np.ndarray:
    if isinstance(obs, np.ndarray):
        return obs
    return np.asarray(obs)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
