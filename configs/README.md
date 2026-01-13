# Configs

These JSON files capture the baseline DQN settings drawn from the referenced DQN papers:
- Experience replay + target network updates
- 84x84 grayscale preprocessing with 4-frame stacks
- Epsilon-greedy exploration (1.0 -> 0.1)
- Reward clipping to [-1, 1] for Atari

Adjust `env_id` to match your Gymnasium/ALE installation.
