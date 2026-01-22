# Training Loop Notes (2026-01-22)

Collected observations to address before the next training session:

## Mismatches
- `dqn.optimizer` config key is ignored; trainer always builds `torch.optim.RMSprop` in `_build_agent`.
- DQN-level `reward_clip` flag (present in CartPole configs) is unused; reward clipping only follows `preprocess.reward_clip`.

## Stability TODOs
- Consider optional gradient clipping (e.g., max-norm 10.0) inside `DQNAgent.update` to tame rare PER + NoisyNet spikes.

## Validation steps
- Smoke test CartPole on CPU for ~10k frames to verify logging/checkpoint flow: activate venv, run `python train/train_cartpole.py --config configs/cartpole_m1_fast.json` (or a temp config with fewer frames). Confirm `reports/metrics_CartPole-v1.csv` and `models/CartPole-v1_latest.pt` appear.
- For Atari, ensure ALE ROMs are installed; run `python train/train_atari.py --config configs/space_invaders_m1_fast.json` and inspect `reports/events_ALE_SpaceInvaders-v5.jsonl` when `log_events` is enabled.

## Next actions
1) Decide whether to honor or drop `dqn.optimizer`; if honoring, add an optimizer factory keyed by the config.
2) Align reward clipping flag location (move to `preprocess` or wire the current DQN key through the loop).
3) Add config knob for gradient clipping and default it off.
