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

Rainbow-Lite switches:
- `dueling`: enable dueling value/advantage heads.
- `noisy`: enable NoisyLinear exploration (epsilon still supported but optional).
- `n_step`: multi-step returns.
- `per_alpha`, `per_beta_start`, `per_beta_frames`: prioritized replay settings.
- `efficient_replay`: store single frames and reconstruct stacks to reduce RAM usage.
