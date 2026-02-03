# Training Loop Notes (2026-01-22)

Collected observations to address before the next training session:

## Mismatches (RESOLVED)
- ~~`dqn.optimizer` config key is ignored~~ - Fixed: trainer now supports "rmsprop", "adam", "sgd"
- ~~DQN-level `reward_clip` flag is unused~~ - Fixed: trainer checks both `dqn.reward_clip` and `preprocess.reward_clip`

## Stability TODOs
- Consider optional gradient clipping (e.g., max-norm 10.0) inside `DQNAgent.update` to tame rare PER + NoisyNet spikes.

## Validation steps
- Smoke test CartPole on CPU for ~10k frames to verify logging/checkpoint flow: activate venv, run `python train/train_cartpole.py --config configs/cartpole_m1_fast.json` (or a temp config with fewer frames). Confirm `reports/metrics_CartPole-v1.csv` and `models/CartPole-v1_latest.pt` appear.
- For Atari, ensure ALE ROMs are installed; run `python train/train_atari.py --config configs/space_invaders_m1_fast.json` and inspect `reports/events_ALE_SpaceInvaders-v5.jsonl` when `log_events` is enabled.

## Next actions
1) Decide whether to honor or drop `dqn.optimizer`; if honoring, add an optimizer factory keyed by the config.
2) Align reward clipping flag location (move to `preprocess` or wire the current DQN key through the loop).
3) Add config knob for gradient clipping and default it off.

## Pacman status (2026-01-23)
- `reports/metrics_ALE_MsPacman-v5.csv` shows training started and reached ~52,508 frames (episode 287) with epsilon ~0.9055.
- `reports/pacman.log` and `reports/pacman_m1_fast.log` only show ALE startup plus OpenCV duplicate-library warnings; no crash stack traces recorded.
- Checkpoint present at `models/ALE_MsPacman-v5_latest.pt`.

## Pacman restart (2026-01-22 21:28:24)
- Archived prior run artifacts to `reports/archive/pacman_20260122_204424` and `models/archive/pacman_20260122_204424`.
- Plan: restart all three Pacman configs (`pacman_m1_fast.json`, `pacman_m1_turbo.json`, `pacman_m1_max.json`) in parallel with caffeinate + max CPU threads.

## Pacman restart launch (2026-01-22 21:30:38)
- Launched all three configs in parallel under `reports/pacman_runs/20260122_213038` with isolated `reports/` + `models/` per run.
- CPU thread env set to 10 (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS`).
- Wrapped the run in `caffeinate -dimsu` so it stops automatically when the training processes exit.
- `pmset highpower` and negative `nice` were attempted but require sudo password; not applied.

## Pacman restart launch (2026-01-22 21:41:04)
- Running all three configs in parallel inside tmux session `pacman_all_20260122_214104`.
- Run root: `reports/pacman_runs/20260122_214104` (logs per run in `fast/`, `turbo/`, `max/`).
- Active PIDs: fast=35870, turbo=35871, max=35872.
- `caffeinate -dimsu` wrapped the script so it stops automatically when training exits.
- Update (2026-01-22 23:16:50): Killed fast and turbo so we can document the different approaches.
- Status at stop:
  * fast metrics last row: frame 949,419 / episode 3,133 / reward 52.
  * turbo metrics last row: frame 465,759 / episode 1,313 / reward 51.
  * max already finished earlier (metrics checkpoint saved at ~2.6M frames).
