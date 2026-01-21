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
from .utils import LinearSchedule, configure_threads, ensure_dir, get_device, seed_everything, to_numpy


@dataclass
class TrainResult:
    metrics_path: Path
    checkpoint_path: Path


@dataclass
class PlateauState:
    prev_eval_returns: list[float] | None = None
    prev_action_dist: np.ndarray | None = None
    prev_adv_gap: float | None = None
    prev_td_error: float | None = None
    perf_hits: int = 0
    gate_hits: int = 0
    last_action_frame: int = 0


def _write_metrics_header(path: Path) -> None:
    path.write_text("frame,episode,episode_reward,epsilon,loss\n")


def _append_metrics(path: Path, frame: int, episode: int, reward: float, epsilon: float, loss: float) -> None:
    with path.open("a") as f:
        f.write(f"{frame},{episode},{reward:.3f},{epsilon:.4f},{loss:.6f}\n")


def _log_event(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    payload = dict(payload)
    payload.setdefault("timestamp", time.time())
    with path.open("a") as f:
        f.write(json.dumps(payload) + "\n")


def _bootstrap_p_improve(
    current: list[float],
    previous: list[float],
    delta: float,
    samples: int,
    rng: np.random.Generator,
) -> float:
    if not current or not previous:
        return 1.0
    cur = np.array(current, dtype=np.float32)
    prev = np.array(previous, dtype=np.float32)
    cur_mean = cur.mean()
    prev_mean = prev.mean()
    if cur_mean > prev_mean + delta:
        return 1.0
    hits = 0
    for _ in range(samples):
        cur_bs = rng.choice(cur, size=cur.size, replace=True).mean()
        prev_bs = rng.choice(prev, size=prev.size, replace=True).mean()
        if cur_bs > prev_bs + delta:
            hits += 1
    return hits / float(samples)


def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    p = np.clip(p.astype(np.float64), eps, 1.0)
    q = np.clip(q.astype(np.float64), eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def _action_distribution(q_values: torch.Tensor) -> np.ndarray:
    actions = torch.argmax(q_values, dim=1).cpu().numpy()
    num_actions = int(q_values.shape[1])
    counts = np.bincount(actions, minlength=num_actions).astype(np.float64)
    return counts / max(1, counts.sum())


def _advantage_gap(q_values: torch.Tensor) -> float:
    top2 = torch.topk(q_values, k=2, dim=1).values
    gaps = (top2[:, 0] - top2[:, 1]).detach().cpu().numpy()
    return float(np.median(gaps)) if gaps.size else 0.0


def _sample_probe_states(buffer, num_states: int, device: torch.device):
    if len(buffer) < max(2, num_states):
        return None
    if isinstance(buffer, (PrioritizedReplayBuffer, PrioritizedFrameStackReplayBuffer)):
        sample = buffer.sample(num_states, device, beta=1.0)
    else:
        sample = buffer.sample(num_states, device)
    return sample.obs


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


def train_from_config(
    cfg: dict[str, Any],
    output_dir: str = "reports",
    checkpoint_dir: str = "models",
) -> TrainResult:
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    training_cfg = cfg.get("training", {})
    configure_threads(
        training_cfg.get("num_threads"),
        training_cfg.get("interop_threads"),
    )

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
    update_every = int(cfg["training"].get("update_every_frames", 1))
    reward_clip = bool(cfg.get("preprocess", {}).get("reward_clip", False))
    early_stop_reward = cfg["training"].get("early_stop_avg_reward")
    early_stop_min_frames = int(cfg["training"].get("early_stop_min_frames", 0))
    early_stop_eval_windows = int(cfg["training"].get("early_stop_eval_windows", 1))
    early_stop_hits = 0
    td_error_ema = None
    log_every = int(cfg["training"].get("log_every_frames", 0) or 0)
    event_log_path = None
    if cfg["training"].get("log_events", False):
        default_path = Path(output_dir) / f"events_{cfg['env_id'].replace('/', '_')}.jsonl"
        event_log_path = Path(cfg["training"].get("event_log_path", default_path))
        ensure_dir(str(event_log_path.parent))
    plateau_enabled = bool(cfg["training"].get("plateau_gate_enabled", False))
    plateau_min_frames = int(cfg["training"].get("plateau_min_frames", 0))
    plateau_eval_windows = int(cfg["training"].get("plateau_eval_windows", 2))
    plateau_perf_delta = float(cfg["training"].get("plateau_perf_delta", 1.0))
    plateau_perf_p_threshold = float(cfg["training"].get("plateau_perf_p_threshold", 0.2))
    plateau_bootstrap_samples = int(cfg["training"].get("plateau_bootstrap_samples", 200))
    plateau_churn_threshold = float(cfg["training"].get("plateau_churn_threshold", 0.01))
    plateau_gap_delta = float(cfg["training"].get("plateau_gap_delta", 0.0))
    plateau_td_improve_threshold = float(cfg["training"].get("plateau_td_improve_threshold", 0.0))
    plateau_probe_states = int(cfg["training"].get("plateau_probe_states", 256))
    td_error_ema_alpha = float(cfg["training"].get("td_error_ema_alpha", 0.05))
    plateau_action = str(cfg["training"].get("plateau_action", "stop"))
    plateau_stop_after_action = bool(cfg["training"].get("plateau_stop_after_action", True))
    plateau_action_cooldown = int(cfg["training"].get("plateau_action_cooldown_frames", 0))
    plateau_epsilon_boost = float(cfg["training"].get("plateau_epsilon_boost", 0.1))
    plateau_boost_frames = int(cfg["training"].get("plateau_boost_frames", 0))
    plateau_lr_decay = float(cfg["training"].get("plateau_lr_decay", 0.5))
    progress_targets_raw = cfg["training"].get("progress_targets", [])
    progress_targets = []
    if isinstance(progress_targets_raw, list):
        for item in progress_targets_raw:
            if not isinstance(item, dict):
                continue
            if "frame" not in item or "min_score" not in item:
                continue
            try:
                progress_targets.append(
                    {
                        "frame": int(item["frame"]),
                        "min_score": float(item["min_score"]),
                        "label": str(item.get("label", "")),
                    }
                )
            except (TypeError, ValueError):
                continue
    progress_targets.sort(key=lambda x: x["frame"])
    progress_index = 0
    plateau_state = PlateauState()
    rng = np.random.default_rng(seed + 123)
    per_beta_schedule = LinearSchedule(
        start=float(cfg["dqn"].get("per_beta_start", 0.4)),
        end=1.0,
        duration_frames=int(cfg["dqn"].get("per_beta_frames", total_frames)),
    )
    n_step = int(cfg["dqn"].get("n_step", 1))
    n_step_buffer = deque(maxlen=n_step)
    epsilon_boost_until = 0

    ensure_dir(output_dir)
    metrics_path = Path(output_dir) / f"metrics_{cfg['env_id'].replace('/', '_')}.csv"
    _write_metrics_header(metrics_path)

    ensure_dir(checkpoint_dir)
    checkpoint_path = Path(checkpoint_dir) / f"{cfg['env_id'].replace('/', '_')}_latest.pt"

    episode = 0
    episode_reward = 0.0
    losses = []
    last_episode_reward = 0.0
    if log_every:
        _log_event(event_log_path, {"type": "start", "frame": 0, "env_id": cfg["env_id"]})

    for frame in range(1, total_frames + 1):
        agent.reset_noise()
        if frame <= epsilon_boost_until:
            if np.random.rand() < plateau_epsilon_boost:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs, eval_mode=True)
        else:
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
        if frame > learning_starts and len(buffer) >= agent.config.batch_size and frame % update_every == 0:
            beta = per_beta_schedule.value(frame)
            if isinstance(buffer, (PrioritizedReplayBuffer, PrioritizedFrameStackReplayBuffer)):
                batch = buffer.sample(agent.config.batch_size, device, beta=beta)
            else:
                batch = buffer.sample(agent.config.batch_size, device)
            loss, td_errors = agent.update(batch)
            if isinstance(buffer, (PrioritizedReplayBuffer, PrioritizedFrameStackReplayBuffer)):
                buffer.update_priorities(batch.indices, td_errors + 1e-6)
            losses.append(loss)
            td_error_mean = float(np.mean(td_errors)) if td_errors.size else 0.0
            if td_error_ema is None:
                td_error_ema = td_error_mean
            else:
                td_error_ema = (1.0 - td_error_ema_alpha) * td_error_ema + td_error_ema_alpha * td_error_mean
            agent.maybe_update_target()

        if done:
            episode += 1
            avg_loss = float(np.mean(losses)) if losses else 0.0
            _append_metrics(metrics_path, frame, episode, episode_reward, agent.epsilon(), avg_loss)
            last_episode_reward = episode_reward
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
            score, returns = evaluate_agent(
                agent,
                eval_env,
                eval_episodes,
                eval_epsilon,
                cfg.get("observation_type"),
                return_returns=True,
            )
            _log_event(
                event_log_path,
                {
                    "type": "eval",
                    "frame": frame,
                    "episode": episode,
                    "score": score,
                    "epsilon": agent.epsilon(),
                },
            )
            while progress_index < len(progress_targets) and frame >= progress_targets[progress_index]["frame"]:
                target = progress_targets[progress_index]
                passed = score >= target["min_score"]
                _log_event(
                    event_log_path,
                    {
                        "type": "progress_hit" if passed else "progress_miss",
                        "frame": frame,
                        "episode": episode,
                        "score": score,
                        "target_frame": target["frame"],
                        "target_score": target["min_score"],
                        "label": target.get("label", ""),
                    },
                )
                progress_index += 1
            if early_stop_reward is not None and frame >= early_stop_min_frames:
                if score >= float(early_stop_reward):
                    early_stop_hits += 1
                else:
                    early_stop_hits = 0
                if early_stop_hits >= early_stop_eval_windows:
                    _log_event(
                        event_log_path,
                        {
                            "type": "early_stop",
                            "frame": frame,
                            "episode": episode,
                            "score": score,
                        },
                    )
                    break

            if plateau_enabled and frame >= plateau_min_frames:
                if plateau_state.prev_eval_returns is None:
                    plateau_state.prev_eval_returns = returns
                else:
                    p_improve = _bootstrap_p_improve(
                        returns,
                        plateau_state.prev_eval_returns,
                        plateau_perf_delta,
                        plateau_bootstrap_samples,
                        rng,
                    )
                    perf_plateau = p_improve < plateau_perf_p_threshold
                    if perf_plateau:
                        plateau_state.perf_hits += 1
                    else:
                        plateau_state.perf_hits = 0

                    probe = _sample_probe_states(buffer, plateau_probe_states, device)
                    churn_ok = True
                    gap_ok = True
                    td_ok = True
                    churn = None
                    gap = None
                    if probe is not None:
                        with torch.no_grad():
                            q_vals = agent.online_net(probe)
                        action_dist = _action_distribution(q_vals)
                        if plateau_state.prev_action_dist is not None:
                            churn = _js_divergence(action_dist, plateau_state.prev_action_dist)
                            churn_ok = churn < plateau_churn_threshold
                        plateau_state.prev_action_dist = action_dist

                        gap = _advantage_gap(q_vals)
                        if plateau_state.prev_adv_gap is not None:
                            gap_ok = (gap - plateau_state.prev_adv_gap) <= plateau_gap_delta
                        plateau_state.prev_adv_gap = gap

                    if td_error_ema is not None and plateau_state.prev_td_error is not None:
                        td_improve = plateau_state.prev_td_error - td_error_ema
                        td_ok = td_improve <= plateau_td_improve_threshold
                    plateau_state.prev_td_error = td_error_ema

                    behavior_plateau = churn_ok
                    learning_plateau = gap_ok or td_ok

                    if perf_plateau and behavior_plateau and learning_plateau:
                        plateau_state.gate_hits += 1
                    else:
                        plateau_state.gate_hits = 0

                    plateau_state.prev_eval_returns = returns

                    if plateau_state.gate_hits >= plateau_eval_windows:
                        _log_event(
                            event_log_path,
                            {
                                "type": "plateau_gate",
                                "frame": frame,
                                "episode": episode,
                                "p_improve": p_improve,
                                "churn": churn,
                                "adv_gap": gap,
                                "td_error_ema": td_error_ema,
                            },
                        )
                        action_cooldown_ok = (
                            plateau_action_cooldown == 0
                            or frame - plateau_state.last_action_frame >= plateau_action_cooldown
                        )
                        if plateau_action != "stop" and action_cooldown_ok:
                            if plateau_action == "epsilon_boost":
                                epsilon_boost_until = frame + plateau_boost_frames
                                plateau_state.last_action_frame = frame
                                _log_event(
                                    event_log_path,
                                    {
                                        "type": "plateau_action",
                                        "action": "epsilon_boost",
                                        "frame": frame,
                                        "duration_frames": plateau_boost_frames,
                                        "epsilon": plateau_epsilon_boost,
                                    },
                                )
                            elif plateau_action == "lr_decay":
                                for group in agent.optimizer.param_groups:
                                    group["lr"] *= plateau_lr_decay
                                plateau_state.last_action_frame = frame
                                _log_event(
                                    event_log_path,
                                    {
                                        "type": "plateau_action",
                                        "action": "lr_decay",
                                        "frame": frame,
                                        "lr_decay": plateau_lr_decay,
                                    },
                                )
                            plateau_state.gate_hits = 0
                        if plateau_stop_after_action or plateau_action == "stop":
                            break

        if checkpoint_every and frame % checkpoint_every == 0:
            _save_checkpoint(checkpoint_path, agent)

        if log_every and frame % log_every == 0:
            avg_loss = float(np.mean(losses)) if losses else 0.0
            _log_event(
                event_log_path,
                {
                    "type": "heartbeat",
                    "frame": frame,
                    "episode": episode,
                    "epsilon": agent.epsilon(),
                    "avg_loss": avg_loss,
                    "last_reward": last_episode_reward,
                },
            )

    _save_checkpoint(checkpoint_path, agent)
    return TrainResult(metrics_path=metrics_path, checkpoint_path=checkpoint_path)


def evaluate_agent(
    agent: DQNAgent,
    env,
    episodes: int,
    epsilon: float,
    observation_type: str | None = None,
    return_returns: bool = False,
) -> float | tuple[float, list[float]]:
    was_training = agent.online_net.training
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
    if was_training:
        agent.online_net.train()
    mean_score = float(np.mean(rewards)) if rewards else 0.0
    if return_returns:
        return mean_score, rewards
    return mean_score


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
