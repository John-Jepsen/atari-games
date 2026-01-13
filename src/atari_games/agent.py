from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .replay import ReplayBuffer, ReplaySample
from .utils import EpsilonSchedule


@dataclass
class AgentConfig:
    gamma: float
    batch_size: int
    target_update_steps: int
    learning_rate: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_frames: int


class DQNAgent:
    def __init__(
        self,
        online_net: nn.Module,
        target_net: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: AgentConfig,
        device: torch.device,
    ) -> None:
        self.online_net = online_net
        self.target_net = target_net
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.eps_schedule = EpsilonSchedule(
            start=config.epsilon_start,
            end=config.epsilon_end,
            decay_frames=config.epsilon_decay_frames,
        )
        self.steps_done = 0
        self.loss_fn = nn.SmoothL1Loss()
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

    def select_action(self, obs: np.ndarray, eval_mode: bool = False) -> int:
        eps = 0.0 if eval_mode else self.eps_schedule.value(self.steps_done)
        if np.random.rand() < eps:
            return -1
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(obs_t)
        return int(torch.argmax(q_values, dim=1).item())

    def update(self, batch: ReplaySample) -> float:
        q_values = self.online_net(batch.obs).gather(1, batch.actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(batch.next_obs).max(1)[0]
            target = batch.rewards + self.config.gamma * (1.0 - batch.dones) * next_q
        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def maybe_update_target(self) -> None:
        if self.steps_done % self.config.target_update_steps == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

    def step(self) -> None:
        self.steps_done += 1

    def epsilon(self) -> float:
        return self.eps_schedule.value(self.steps_done)


def choose_action(agent: DQNAgent, obs: np.ndarray, action_space) -> int:
    action = agent.select_action(obs)
    if action == -1:
        return action_space.sample()
    return action
