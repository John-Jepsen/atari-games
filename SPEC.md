# Atari Games Project Spec (Step 1)

## Purpose
Build three reinforcement-learning agents that play CartPole, Space Invaders, and Pacman using deep neural networks. The approach should align with the provided project brief and the referenced deep learning / DQN papers.

## Required Inputs
- Project brief: `atari-games-instructions/atari-games.md`.
- Reference docs in `doc-referances/` (all PDFs): foundational deep learning overview, list of DL models, and two DQN/Atari papers.

## Scope and Deliverables
- **Models**: one agent per game (CartPole, Space Invaders, Pacman).
- **Training code**: reusable DQN components, preprocessing, and training/eval entry points.
- **Report**: a short blog post explaining the approach, experiments, and results.

## Success Criteria (initial)
- Agents learn from raw observations and rewards only (no game-specific features beyond action space).
- Demonstrated improvement over random policy for each game.
- Reproducible training runs (fixed seeds, logged configs, and environment versions).

## Reference-Driven Design Requirements
- **RL framing**: maximize cumulative reward via reinforcement learning (Intro DL doc).
- **Image input**: use convolutional networks for visual data (Intro DL + DL models list).
- **DQN baseline**: experience replay, target network updates, and Q-learning with a deep CNN (DQN papers).

## Baseline Algorithm (DQN)
Minimum baseline aligned to the DQN papers:
- **Preprocessing**: convert Atari frames to grayscale, downsample to 110x84, crop to 84x84, and stack 4 frames as input.
- **Network**: CNN with rectifier activations; reference architecture (Nature 2015):
  - Conv 32 @ 8x8 stride 4
  - Conv 64 @ 4x4 stride 2
  - Conv 64 @ 3x3 stride 1
  - FC 512
  - Linear output head per action
- **Replay memory**: uniform sampling from a fixed-size buffer (1e6 transitions).
- **Target network**: copy online weights every C steps (configurable).
- **Optimization**: RMSProp, minibatch size 32.
- **Exploration**: epsilon-greedy with anneal 1.0 -> 0.1 over ~1M frames, then fixed.
- **Reward scaling**: clip rewards to [-1, 1] during training (per DQN papers).
- **Discount**: gamma 0.99.
- **Frame skip**: k=4 default; adjust if visibility issues occur (e.g., Space Invaders).

## Evaluation
- Report average score per episode across multiple evaluation runs.
- Use a fixed evaluation epsilon (e.g., 0.05) to reduce variance.
- Record training curves for episode return and predicted Q values (DQN guidance).

## Reproducibility and Reporting
- Log hyperparameters, seeds, and environment versions for each run.
- Save only light artifacts by default (configs, metrics, plots); keep large checkpoints out of Git.
- The blog post should include: model design choices, training details, key results, and limitations.
