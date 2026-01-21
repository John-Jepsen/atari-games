# Pac-Man Max-Speed Plan (M1)

## Goal
Push the M1 as hard as possible for Ms. Pac-Man while keeping results reliable enough to be called strong.

## Hardware + OS
- Run **one model at a time** (avoid CPU contention).
- Keep **caffeinate** running to prevent sleep.
- Plug in power, keep thermal headroom (lid open, airflow).

## Compute + Threading
- Explicitly set CPU threading:
  - `torch.set_num_threads(N)`
  - `torch.set_num_interop_threads(M)`
- Env vars: `OMP_NUM_THREADS`, `MKL_NUM_THREADS`
- Use **MPS** for this max-speed profile; fall back to CPU if MPS proves slower.
- Use config: `configs/pacman_m1_max.json`.

## Training throughput (speed)
- `update_every_frames = 4` (fewer optimizer steps)
- `eval_every_frames = 200000` (reduce eval overhead)
- `eval_episodes = 5`
- `checkpoint_every_frames = 200000`
- `log_every_frames = 20000`
- `efficient_replay = true`
- Reduced replay capacity to 300,000 to lower RAM pressure.

## Learning efficiency (accuracy)
- Dueling DQN + n-step (PER + NoisyNet disabled to speed up)
- Reward clipping for Atari
- Frame skip = 4, grayscale, 84x84, stack 4 frames

## Plateau gate + auto actions
- Multi-objective gate (performance + behavior + learning)
- Action on plateau: **epsilon boost** (try to escape local optimum)
- Optional fallback: LR decay

## Early progress checkpoints (signals before 2 hours)
- 200k frames: min score 5
- 500k frames: min score 20
- 1.0M frames: min score 50
- 1.5M frames: min score 100
- 2.0M frames: min score 200

## Strong stop targets
- Early stop if avg eval score >= 1000 (2 consecutive evals, after 2M frames)

## Risks / Tradeoffs
- More updates can reduce FPS but usually improves sample efficiency.
- Thread settings need tuning per machine to avoid oversubscription.
- Too aggressive early-stop or plateau settings can halt a run prematurely.
