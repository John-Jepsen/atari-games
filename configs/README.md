# Configs

These JSON files capture the baseline DQN settings drawn from the referenced DQN papers:
- Experience replay + target network updates
- 84x84 grayscale preprocessing with 4-frame stacks
- Epsilon-greedy exploration (1.0 -> 0.1)
- Reward clipping to [-1, 1] for Atari

Training keys include `learning_starts` and `checkpoint_every_frames` for warm-up and periodic saves.
Adjust `env_id` to match your Gymnasium/ALE installation.

Device control:
- `device`: set to `mps`, `cpu`, or `auto` (default). M1 users can force `mps` for GPU acceleration.

Fast M1 configs:
- `*_m1_fast.json` favor wall-clock speed on Apple Silicon. They default to `cpu` because
  local benchmarks showed higher training FPS on CPU vs MPS for these workloads.

Turbo configs:
- `*_m1_turbo.json` enable NoisyNet exploration and update every frame for faster learning.
  They include optional early-stop thresholds for strong performance targets.

Plateau gate (multi-objective) keys:
- `plateau_gate_enabled`: enable multi-signal plateau detection.
- `plateau_min_frames`: wait before checks begin.
- `plateau_eval_windows`: consecutive eval windows required to trigger.
- `plateau_perf_delta`: minimum meaningful return improvement (ROPE).
- `plateau_perf_p_threshold`: bootstrap P(improve) threshold.
- `plateau_bootstrap_samples`: bootstrap resample count.
- `plateau_churn_threshold`: JS divergence threshold for policy churn.
- `plateau_gap_delta`: advantage-gap improvement threshold.
- `plateau_td_improve_threshold`: TD-error improvement threshold.
- `plateau_probe_states`: number of probe states sampled from replay.
- `td_error_ema_alpha`: smoothing factor for TD-error EMA.
- `plateau_action`: `stop`, `epsilon_boost`, or `lr_decay`.
- `plateau_stop_after_action`: stop immediately after action if true.
- `plateau_action_cooldown_frames`: minimum frames between actions.
- `plateau_epsilon_boost`: exploration rate during boost.
- `plateau_boost_frames`: duration of exploration boost.
- `plateau_lr_decay`: LR multiplier when decay action triggers.

Event logging:
- `log_events`: write JSONL events to `reports/events_<env>.jsonl`.
- `log_every_frames`: add heartbeat events every N frames.

Rainbow-Lite switches:
- `dueling`: enable dueling value/advantage heads.
- `noisy`: enable NoisyLinear exploration (epsilon still supported but optional).
- `n_step`: multi-step returns.
- `per_alpha`, `per_beta_start`, `per_beta_frames`: prioritized replay settings.
- `efficient_replay`: store single frames and reconstruct stacks to reduce RAM usage.
