from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .agent import AgentConfig, DQNAgent, choose_action
from .envs import make_atari, make_cartpole
from .networks import DQNCNN, DQNMLP, DuelingDQNCNN, DuelingDQNMLP
from .preprocess import format_obs
from .replay import (
    FrameStackReplayBuffer,
    PrioritizedFrameStackReplayBuffer,
    PrioritizedReplayBuffer,
    ReplayBuffer,
)
from .utils import LinearSchedule, ensure_dir, get_device, seed_everything, to_numpy


@dataclass
class TrainResult:
    metrics_path: Path
    checkpoint_path: Path


def _write_metrics_header(path: Path) -> None:
    path.write_text("frame,episode,episode_reward,epsilon,loss\n")


def _append_metrics(path: Path, frame: int, episode: int, reward: float, epsilon: float, loss: float) -> None:
    with path.open("a") as f:
        f.write(f"{frame},{episode},{reward:.3f},{epsilon:.4f},{loss:.6f}\n")


def _build_agent(cfg: dict[str, Any], obs_shape: tuple[int, ...], num_actions: int, device: torch.device) -> DQNAgent:
    dqn_cfg = cfg["dqn"]
    agent_cfg = AgentConfig(
        gamma=dqn_cfg["gamma"],
        batch_size=dqn_cfg["batch_size"],
        target_update_steps=dqn_cfg["target_update_steps"],
        learning_rate=dqn_cfg["learning_rate"],
        epsilon_start=dqn_cfg["epsilon_start"],
        epsilon_end=dqn_cfg["epsilon_end"],
        epsilon_decay_frames=dqn_cfg["epsilon_decay_frames"],
        n_step=int(dqn_cfg.get("n_step", 1)),
        noisy=bool(dqn_cfg.get("noisy", False)),
    )

    if cfg.get("observation_type") == "pixels":
        input_channels = obs_shape[0]
        if dqn_cfg.get("dueling", True):
            online = DuelingDQNCNN(input_channels=input_channels, num_actions=num_actions, noisy=agent_cfg.noisy).to(device)
            target = DuelingDQNCNN(input_channels=input_channels, num_actions=num_actions, noisy=agent_cfg.noisy).to(device)
        else:
            online = DQNCNN(input_channels=input_channels, num_actions=num_actions).to(device)
            target = DQNCNN(input_channels=input_channels, num_actions=num_actions).to(device)
    else:
        input_dim = obs_shape[0]
        if dqn_cfg.get("dueling", True):
            online = DuelingDQNMLP(input_dim=input_dim, num_actions=num_actions, noisy=agent_cfg.noisy).to(device)
            target = DuelingDQNMLP(input_dim=input_dim, num_actions=num_actions, noisy=agent_cfg.noisy).to(device)
        else:
            online = DQNMLP(input_dim=input_dim, num_actions=num_actions).to(device)
            target = DQNMLP(input_dim=input_dim, num_actions=num_actions).to(device)

    optimizer = torch.optim.RMSprop(online.parameters(), lr=agent_cfg.learning_rate)
    return DQNAgent(online, target, optimizer, agent_cfg, device)


def _prepare_buffer(cfg: dict[str, Any], obs_shape: tuple[int, ...]) -> ReplayBuffer:
    dqn_cfg = cfg["dqn"]
    obs_dtype = np.uint8 if cfg.get("observation_type") == "pixels" else np.float32
    if cfg.get("observation_type") == "pixels" and dqn_cfg.get("efficient_replay", False):
        frame_shape = obs_shape[1:] if len(obs_shape) == 3 else obs_shape
        stack_size = int(cfg.get("preprocess", {}).get("frame_stack", 4))
        if dqn_cfg.get("per_alpha", 0.0) > 0.0:
            return PrioritizedFrameStackReplayBuffer(
                dqn_cfg["replay_capacity"],
                frame_shape=frame_shape,
                stack_size=stack_size,
                alpha=float(dqn_cfg.get("per_alpha", 0.6)),
            )
        return FrameStackReplayBuffer(
            dqn_cfg["replay_capacity"], frame_shape=frame_shape, stack_size=stack_size
        )

    if dqn_cfg.get("per_alpha", 0.0) > 0.0:
        return PrioritizedReplayBuffer(
            dqn_cfg["replay_capacity"],
            obs_shape=obs_shape,
            obs_dtype=obs_dtype,
            alpha=float(dqn_cfg.get("per_alpha", 0.6)),
        )
    return ReplayBuffer(dqn_cfg["replay_capacity"], obs_shape=obs_shape, obs_dtype=obs_dtype)


def _save_checkpoint(path: Path, agent: DQNAgent) -> None:
    payload = {
        "model_state": agent.online_net.state_dict(),
        "steps_done": agent.steps_done,
    }
    torch.save(payload, path)


def train_from_config(cfg: dict[str, Any], output_dir: str = "reports") -> TrainResult:
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    device = get_device(cfg.get("device"))

    if cfg.get("observation_type") == "pixels":
        env = make_atari(cfg["env_id"], seed=seed, preprocess=cfg["preprocess"])
    else:
        env = make_cartpole(cfg["env_id"], seed=seed)

    eval_env = None
    if cfg.get("training", {}).get("eval_episodes"):
        if cfg.get("observation_type") == "pixels":
            eval_env = make_atari(cfg["env_id"], seed=seed + 1, preprocess=cfg["preprocess"])
        else:
            eval_env = make_cartpole(cfg["env_id"], seed=seed + 1)

    obs, _ = env.reset()
    obs = format_obs(to_numpy(obs), cfg.get("observation_type"))
    obs_shape = obs.shape
    num_actions = env.action_space.n

    agent = _build_agent(cfg, obs_shape, num_actions, device)
    buffer = _prepare_buffer(cfg, obs_shape)

    total_frames = int(cfg["training"]["total_frames"])
    eval_every = int(cfg["training"].get("eval_every_frames", 0) or 0)
    eval_episodes = int(cfg["training"].get("eval_episodes", 0) or 0)
    eval_epsilon = float(cfg["training"].get("eval_epsilon", 0.05))
    learning_starts = int(cfg["training"].get("learning_starts", 1000))
    checkpoint_every = int(cfg["training"].get("checkpoint_every_frames", 0) or 0)
    reward_clip = bool(cfg.get("preprocess", {}).get("reward_clip", False))
    per_beta_schedule = LinearSchedule(
        start=float(cfg["dqn"].get("per_beta_start", 0.4)),
        end=1.0,
        duration_frames=int(cfg["dqn"].get("per_beta_frames", total_frames)),
    )
    n_step = int(cfg["dqn"].get("n_step", 1))
    n_step_buffer = deque(maxlen=n_step)

    ensure_dir(output_dir)
    metrics_path = Path(output_dir) / f"metrics_{cfg['env_id'].replace('/', '_')}.csv"
    _write_metrics_header(metrics_path)

    ensure_dir("models")
    checkpoint_path = Path("models") / f"{cfg['env_id'].replace('/', '_')}_latest.pt"

    episode = 0
    episode_reward = 0.0
    losses = []

    for frame in range(1, total_frames + 1):
        agent.reset_noise()
        action = choose_action(agent, obs, env.action_space)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if reward_clip:
            reward = float(np.clip(reward, -1.0, 1.0))
        next_obs = format_obs(to_numpy(next_obs), cfg.get("observation_type"))

        n_step_buffer.append((obs, action, reward, next_obs, done))
        if len(n_step_buffer) == n_step:
            R = 0.0
            done_n = False
            next_obs_n = n_step_buffer[-1][3]
            for i, (_, _, r, _, d) in enumerate(n_step_buffer):
                R += (cfg["dqn"]["gamma"] ** i) * r
                if d:
                    done_n = True
                    break
            obs_0, action_0 = n_step_buffer[0][0], n_step_buffer[0][1]
            buffer.add(obs_0, action_0, R, next_obs_n, done_n)
            n_step_buffer.popleft()
        obs = next_obs
        episode_reward += reward

        agent.step()
        if frame > learning_starts and len(buffer) >= agent.config.batch_size:
            beta = per_beta_schedule.value(frame)
            if isinstance(buffer, (PrioritizedReplayBuffer, PrioritizedFrameStackReplayBuffer)):
                batch = buffer.sample(agent.config.batch_size, device, beta=beta)
            else:
                batch = buffer.sample(agent.config.batch_size, device)
            loss, td_errors = agent.update(batch)
            if isinstance(buffer, (PrioritizedReplayBuffer, PrioritizedFrameStackReplayBuffer)):
                buffer.update_priorities(batch.indices, td_errors + 1e-6)
            losses.append(loss)
            agent.maybe_update_target()

        if done:
            episode += 1
            avg_loss = float(np.mean(losses)) if losses else 0.0
            _append_metrics(metrics_path, frame, episode, episode_reward, agent.epsilon(), avg_loss)
            # Flush remaining n-step transitions (shorter horizons)
            while n_step_buffer:
                R = 0.0
                done_n = False
                next_obs_n = n_step_buffer[-1][3]
                for i, (_, _, r, _, d) in enumerate(n_step_buffer):
                    R += (cfg["dqn"]["gamma"] ** i) * r
                    if d:
                        done_n = True
                        break
                obs_0, action_0 = n_step_buffer[0][0], n_step_buffer[0][1]
                buffer.add(obs_0, action_0, R, next_obs_n, done_n)
                n_step_buffer.popleft()

            obs, _ = env.reset()
            obs = format_obs(to_numpy(obs), cfg.get("observation_type"))
            episode_reward = 0.0
            losses = []

        if eval_env and eval_every and frame % eval_every == 0:
            _ = evaluate_agent(agent, eval_env, eval_episodes, eval_epsilon, cfg.get("observation_type"))

        if checkpoint_every and frame % checkpoint_every == 0:
            _save_checkpoint(checkpoint_path, agent)

    _save_checkpoint(checkpoint_path, agent)
    return TrainResult(metrics_path=metrics_path, checkpoint_path=checkpoint_path)


def evaluate_agent(agent: DQNAgent, env, episodes: int, epsilon: float, observation_type: str | None = None) -> float:
    agent.online_net.eval()
    rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        obs = format_obs(to_numpy(obs), observation_type)
        done = False
        total = 0.0
        while not done:
            action = agent.select_action(obs, eval_mode=True)
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            obs = format_obs(to_numpy(next_obs), observation_type)
            total += reward
        rewards.append(total)
    agent.online_net.train()
    return float(np.mean(rewards)) if rewards else 0.0


def load_checkpoint(path: str, agent: DQNAgent) -> None:
    payload = torch.load(path, map_location=agent.device)
    agent.online_net.load_state_dict(payload["model_state"])
    agent.target_net.load_state_dict(payload["model_state"])
    agent.steps_done = payload.get("steps_done", 0)


def dump_run_summary(cfg: dict[str, Any], result: TrainResult, output_path: str = "reports/run_summary.json") -> None:
    summary = {
        "env_id": cfg["env_id"],
        "metrics_path": str(result.metrics_path),
        "checkpoint_path": str(result.checkpoint_path),
        "config": cfg,
        "timestamp": time.time(),
    }
    Path(output_path).write_text(json.dumps(summary, indent=2))
