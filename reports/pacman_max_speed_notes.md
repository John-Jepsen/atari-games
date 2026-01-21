# Pac-Man Max-Speed Plan (M1)

## Goal
Push the M1 as hard as possible for Ms. Pac-Man while keeping results reliable enough to be called strong.

## Hardware + OS
- Run **one model at a time** (avoid CPU contention).
- Keep **caffeinate** running to prevent sleep.
- Plug in power, keep thermal headroom (lid open, airflow).

## Compute + Threading (to add)
- Explicitly set CPU threading:
  - `torch.set_num_threads(N)`
  - `torch.set_num_interop_threads(M)`
  - Env vars: `OMP_NUM_THREADS`, `MKL_NUM_THREADS`
- Use **CPU** (benchmarks show CPU faster than MPS for this project).

## Training throughput (speed)
- `update_every_frames = 1`
- `eval_every_frames = 100000` (early signal)
- `checkpoint_every_frames = 100000`
- `log_every_frames = 10000`
- `efficient_replay = true`
- Optional: reduce replay capacity (1,000,000 -> 500,000) if memory becomes a bottleneck.

## Learning efficiency (accuracy)
- Dueling DQN + PER + n-step + NoisyNet
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
