from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass
class ReplaySample:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor
    weights: torch.Tensor
    indices: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: Tuple[int, ...], obs_dtype: np.dtype) -> None:
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, *obs_shape), dtype=obs_dtype)
        self.next_obs = np.zeros((self.capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)
        self.pos = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> None:
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = 1.0 if done else 0.0
        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True

    def sample(self, batch_size: int, device: torch.device) -> ReplaySample:
        size = len(self)
        indices = np.random.randint(0, size, size=batch_size)
        obs = torch.from_numpy(self.obs[indices]).to(device)
        next_obs = torch.from_numpy(self.next_obs[indices]).to(device)
        actions = torch.from_numpy(self.actions[indices]).to(device)
        rewards = torch.from_numpy(self.rewards[indices]).to(device)
        dones = torch.from_numpy(self.dones[indices]).to(device)
        weights = torch.ones((batch_size,), dtype=torch.float32, device=device)
        return ReplaySample(
            obs=obs,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs,
            dones=dones,
            weights=weights,
            indices=indices,
        )


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity: int, obs_shape: Tuple[int, ...], obs_dtype: np.dtype, alpha: float) -> None:
        super().__init__(capacity=capacity, obs_shape=obs_shape, obs_dtype=obs_dtype)
        self.alpha = float(alpha)
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        self.max_priority = 1.0

    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> None:
        super().add(obs, action, reward, next_obs, done)
        idx = (self.pos - 1) % self.capacity
        self.priorities[idx] = self.max_priority

    def sample(self, batch_size: int, device: torch.device, beta: float) -> ReplaySample:
        size = len(self)
        if size == 0:
            raise ValueError("Cannot sample from an empty buffer.")
        priorities = self.priorities[:size] ** self.alpha
        probs = priorities / priorities.sum()
        indices = np.random.choice(size, batch_size, p=probs)
        weights = (size * probs[indices]) ** (-beta)
        weights = weights / weights.max()

        obs = torch.from_numpy(self.obs[indices]).to(device)
        next_obs = torch.from_numpy(self.next_obs[indices]).to(device)
        actions = torch.from_numpy(self.actions[indices]).to(device)
        rewards = torch.from_numpy(self.rewards[indices]).to(device)
        dones = torch.from_numpy(self.dones[indices]).to(device)
        weights_t = torch.from_numpy(weights.astype(np.float32)).to(device)

        return ReplaySample(
            obs=obs,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs,
            dones=dones,
            weights=weights_t,
            indices=indices,
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        priorities = priorities.astype(np.float32)
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, float(priorities.max()))


class FrameStackReplayBuffer:
    def __init__(self, capacity: int, frame_shape: Tuple[int, ...], stack_size: int) -> None:
        self.capacity = int(capacity)
        self.stack_size = int(stack_size)
        self.frames = np.zeros((self.capacity, *frame_shape), dtype=np.uint8)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)
        self.pos = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def _frame_from_obs(self, obs: np.ndarray) -> np.ndarray:
        if obs.ndim == 3:
            return obs[-1]
        return obs

    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> None:
        self.frames[self.pos] = self._frame_from_obs(obs)
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = 1.0 if done else 0.0
        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True

    def _is_valid_index(self, idx: int) -> bool:
        if self.full:
            invalid = (self.pos - 1) % self.capacity
            return idx != invalid
        return idx < self.pos - 1

    def _encode_observation(self, idx: int) -> np.ndarray:
        stack = np.zeros((self.stack_size, *self.frames.shape[1:]), dtype=np.uint8)
        for i in range(self.stack_size):
            frame_idx = idx - (self.stack_size - 1 - i)
            if not self.full and frame_idx < 0:
                continue
            buf_idx = frame_idx % self.capacity
            stack[i] = self.frames[buf_idx]
            if self.dones[buf_idx]:
                stack[:i] = 0
        return stack

    def sample(self, batch_size: int, device: torch.device) -> ReplaySample:
        size = len(self)
        if size < 2:
            raise ValueError("Not enough samples for frame-stack replay.")
        indices = []
        while len(indices) < batch_size:
            idx = np.random.randint(0, size if self.full else self.pos - 1)
            if self._is_valid_index(idx):
                indices.append(idx)
        indices = np.array(indices, dtype=np.int64)

        obs = np.stack([self._encode_observation(i) for i in indices])
        next_obs = np.stack([self._encode_observation(i + 1) for i in indices])
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        dones = self.dones[indices]

        obs_t = torch.from_numpy(obs).to(device)
        next_obs_t = torch.from_numpy(next_obs).to(device)
        actions_t = torch.from_numpy(actions).to(device)
        rewards_t = torch.from_numpy(rewards).to(device)
        dones_t = torch.from_numpy(dones).to(device)
        weights_t = torch.ones((batch_size,), dtype=torch.float32, device=device)

        return ReplaySample(
            obs=obs_t,
            actions=actions_t,
            rewards=rewards_t,
            next_obs=next_obs_t,
            dones=dones_t,
            weights=weights_t,
            indices=indices,
        )


class PrioritizedFrameStackReplayBuffer(FrameStackReplayBuffer):
    def __init__(self, capacity: int, frame_shape: Tuple[int, ...], stack_size: int, alpha: float) -> None:
        super().__init__(capacity=capacity, frame_shape=frame_shape, stack_size=stack_size)
        self.alpha = float(alpha)
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        self.max_priority = 1.0

    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> None:
        super().add(obs, action, reward, next_obs, done)
        idx = (self.pos - 1) % self.capacity
        self.priorities[idx] = self.max_priority

    def sample(self, batch_size: int, device: torch.device, beta: float) -> ReplaySample:
        size = len(self)
        if size < 2:
            raise ValueError("Not enough samples for frame-stack replay.")
        priorities = self.priorities.copy()
        if self.full:
            invalid = (self.pos - 1) % self.capacity
            priorities[invalid] = 0.0
        else:
            priorities[self.pos - 1 :] = 0.0

        scaled = priorities[: self.capacity] ** self.alpha
        probs = scaled / scaled.sum()
        indices = np.random.choice(self.capacity, batch_size, p=probs)
        weights = (size * probs[indices]) ** (-beta)
        weights = weights / weights.max()

        obs = np.stack([self._encode_observation(i) for i in indices])
        next_obs = np.stack([self._encode_observation(i + 1) for i in indices])
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        dones = self.dones[indices]

        obs_t = torch.from_numpy(obs).to(device)
        next_obs_t = torch.from_numpy(next_obs).to(device)
        actions_t = torch.from_numpy(actions).to(device)
        rewards_t = torch.from_numpy(rewards).to(device)
        dones_t = torch.from_numpy(dones).to(device)
        weights_t = torch.from_numpy(weights.astype(np.float32)).to(device)

        return ReplaySample(
            obs=obs_t,
            actions=actions_t,
            rewards=rewards_t,
            next_obs=next_obs_t,
            dones=dones_t,
            weights=weights_t,
            indices=indices,
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        priorities = priorities.astype(np.float32)
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, float(priorities.max()))
