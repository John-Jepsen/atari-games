# Teaching Neural Networks to Play Atari: A Deep Q-Learning Journey

## Introduction

Can a neural network learn to play video games from scratch, using only raw pixels and a score? This project explores that question by implementing Deep Q-Network (DQN) agents for three classic games: CartPole, Space Invaders, and Pac-Man.

The challenge is significant: unlike traditional game AI that relies on hand-crafted rules and domain knowledge, our agents must discover effective strategies purely through trial and error, learning from millions of gameplay frames.

## The DQN Algorithm

Deep Q-Networks, introduced by DeepMind in 2013-2015, combine deep learning with reinforcement learning. The core idea is elegant: train a neural network to predict the expected future reward (Q-value) for each possible action, then always choose the action with the highest predicted value.

### Key Components

**Experience Replay**: Instead of learning from consecutive frames (which are highly correlated), we store transitions in a replay buffer and sample random batches. This breaks temporal correlations and improves learning stability.

**Target Network**: We maintain two copies of our network: an "online" network that we update frequently, and a "target" network that we update slowly. This prevents the moving target problem where our predictions chase themselves in circles.

**Double DQN**: A refinement where the online network selects actions, but the target network evaluates them. This reduces the overestimation bias inherent in standard Q-learning.

### Network Architecture

For visual games (Space Invaders, Pac-Man), we use a convolutional neural network:
- Input: 4 stacked grayscale frames (84x84 pixels)
- Conv layers: 32@8x8 stride 4 -> 64@4x4 stride 2 -> 64@3x3 stride 1
- Fully connected: 512 units
- Output: Q-values for each action

For CartPole (which uses numeric state observations), we use a simpler MLP with two hidden layers.

## Implementation Details

### Preprocessing Pipeline

Raw Atari frames undergo several transformations:
1. Convert to grayscale (reduces 3 channels to 1)
2. Downsample from 210x160 to 84x84
3. Stack 4 consecutive frames (provides motion information)
4. Normalize pixel values to [0, 1]

Reward clipping to [-1, +1] helps stabilize training across games with different score scales.

### Exploration vs Exploitation

We use epsilon-greedy exploration: with probability epsilon, take a random action; otherwise, take the best action according to our network. Epsilon starts at 1.0 (fully random) and decays to 0.1 over the first million frames.

We also implemented **NoisyNet** as an alternative: adding learned noise to network weights provides state-dependent exploration that can be more efficient than epsilon-greedy.

### Additional Enhancements

- **Prioritized Experience Replay**: Sample transitions with high TD-error more frequently
- **Dueling Architecture**: Separate value and advantage streams in the network
- **n-step Returns**: Bootstrap from rewards n steps ahead instead of just 1

## Results

### CartPole

CartPole is a simple balancing task: keep a pole upright on a moving cart. Our agent reliably learns to balance the pole for extended periods.

- **Frames trained**: 200,000
- **Episodes**: 1,655
- **Performance**: Scores of 22-262 (max is 500)

The agent learns the basic balancing policy quickly, though performance varies episode to episode.

### Space Invaders

Space Invaders requires shooting descending aliens while dodging their projectiles.

- **Frames trained**: 2.4 million
- **Episodes**: 4,326
- **Performance**: Scores of 23-35

The agent learns to shoot aliens but struggles with optimal positioning and timing. Performance is modest compared to expert human play, suggesting more training or architectural improvements could help.

### Pac-Man (Ms. Pac-Man)

Pac-Man requires navigating a maze, eating pellets, and avoiding ghosts.

- **Frames trained**: 2.6 million
- **Episodes**: 6,852
- **Performance**: Scores of 68-119

The agent learns basic pellet-eating behavior and shows some ghost-avoidance tendencies. The stochastic ghost behavior in Ms. Pac-Man makes this a challenging environment.

## Lessons Learned

### CPU vs GPU on Apple Silicon

Surprisingly, training on CPU was faster than using Apple's Metal Performance Shaders (MPS) for these relatively small networks. The overhead of moving data to the GPU outweighs the computational benefits at this scale.

### Hyperparameter Sensitivity

Small changes in learning rate, replay buffer size, and update frequency significantly impact results. We found that:
- Smaller replay buffers (100k vs 1M) train faster but may be less stable
- Updating every frame (vs every 4 frames) accelerates learning
- Thread tuning (`OMP_NUM_THREADS=10`) provided meaningful speedups

### The Exploration Challenge

Balancing exploration and exploitation remains difficult. Too little exploration means the agent gets stuck in local optima; too much means it never exploits what it's learned. The plateau detection system we implemented helps identify when learning has stalled.

## Conclusion

Deep Q-Networks can learn to play video games from raw pixels, but achieving strong performance requires careful engineering. Our implementation demonstrates the core DQN algorithm plus several modern improvements, providing a foundation for further experimentation.

Future directions include:
- Training for more frames (10M+) to match published benchmarks
- Implementing Rainbow DQN (combining all improvements)
- Trying other environments in the Atari suite
- Adding recurrent layers (DRQN) for partially observable games

The code is available in this repository, with configurable experiments for easy reproduction.

---

*Built as part of the Qwasar Data Science curriculum.*
