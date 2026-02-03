# Claude Code Instructions

## Project Overview
Deep reinforcement learning project implementing DQN agents for CartPole, Space Invaders, and Pac-Man using PyTorch. Part of Qwasar Data Science curriculum.

## Architecture
```
src/atari_games/     # Core library
  agent.py           # DQNAgent with Double DQN, n-step returns, NoisyNet support
  networks.py        # CNN for Atari, MLP for CartPole
  replay.py          # Prioritized experience replay buffer
  preprocess.py      # Frame stacking, grayscale, reward clipping
  trainer.py         # Training loop with metrics logging
  envs.py            # Environment wrappers
  config.py          # Config loading
  utils.py           # Epsilon schedule, helpers

train/               # Entry points
  train_cartpole.py  # CartPole training script
  train_atari.py     # Atari (Space Invaders, Pac-Man) training

eval/                # Evaluation
  eval_agent.py      # Evaluate trained models

configs/             # Experiment configurations (JSON)
  *_m1_fast.json     # Fast CPU configs for M1/M2 Macs
  *_m1_turbo.json    # Turbo configs (NoisyNet + aggressive updates)
  *_m1_max.json      # Maximum speed configs (thread tuned)

reports/             # Training outputs (metrics CSV, event logs)
models/              # Checkpoints (gitignored)
```

## Commands

### Environment Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Training
```bash
# CartPole
python train/train_cartpole.py --config configs/cartpole_m1_fast.json

# Space Invaders
python train/train_atari.py --config configs/space_invaders_m1_fast.json

# Pac-Man
python train/train_atari.py --config configs/pacman_m1_fast.json
```

### Evaluation
```bash
python eval/eval_agent.py --model models/CartPole-v1_latest.pt --env CartPole-v1
```

## Coding Conventions
- Python 3.14, PEP 8, 4-space indentation
- Type hints on function signatures
- Dataclasses for config objects (see `AgentConfig` in agent.py)
- Torch tensors on device; numpy for environment interaction

## Key Implementation Details
- **Double DQN**: Online net selects actions, target net evaluates (agent.py:65-66)
- **Prioritized Replay**: TD-error based priorities with importance sampling (replay.py)
- **NoisyNet**: Optional noisy linear layers for exploration (networks.py)
- **n-step returns**: Configurable multi-step bootstrapping (trainer.py)
- **Frame stacking**: 4 frames for Atari, single frame for CartPole

## Config Notes
- CPU is faster than MPS for this workload on Apple Silicon
- Thread tuning: `OMP_NUM_THREADS=10` for max throughput
- `preprocess.reward_clip` controls reward clipping (not `dqn.reward_clip`)
- `dqn.optimizer` config key is currently ignored; RMSprop hardcoded

## Testing
No test suite yet. Smoke test with:
```bash
python train/train_cartpole.py --config configs/cartpole_m1_fast.json
# Check: reports/metrics_CartPole-v1.csv and models/CartPole-v1_latest.pt
```

## References
Consult `doc-referances/` PDFs and `atari-games-instructions/atari-games.md` for DQN algorithm details and project requirements.

## Git Workflow
- **Main branch**: `dev` (all work submitted here)
- Feature branches: `pacman-max-speed`, etc.
- Concise imperative commit messages (e.g., "Add gradient clipping option")
- Include reproducibility info in PRs (seeds, configs, results)
- Merge feature branches to `dev` when complete

## Training Results
All three models are trained:

| Game | Frames | Episodes | Reward Range | Checkpoint |
|------|--------|----------|--------------|------------|
| CartPole | 200k | 1,655 | 22-262 | `models/CartPole-v1_latest.pt` |
| Space Invaders | 2.4M | 4,326 | 23-35 | `models/ALE_SpaceInvaders-v5_latest.pt` |
| Pac-Man (max) | 2.6M | 6,852 | 68-119 | `reports/pacman_runs/.../max/models/ALE_MsPacman-v5_latest.pt` |
