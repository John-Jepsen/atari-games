# Benchmark Summary

- run_tag: 20260120_space_invaders
- timestamp: 1768935002.378411
- env_id: ALE/SpaceInvaders-v5
- config_path: configs/space_invaders.json
- config_summary: env_id=ALE/SpaceInvaders-v5, seed=42, gamma=0.99, replay_capacity=1000000, batch_size=32, total_frames=10000000
- frames: 5000
- mode: both

## Results

| device | env_fps | train_fps |
| --- | --- | --- |
| mps | 1890.26 | 162.98 |
| cpu | 1924.03 | 252.34 |

## Recommendation

Best training throughput: cpu (252.34 fps).

## Bottleneck hints

Heuristic based on env_fps vs train_fps (higher ratio means training dominates).
- mps: training-bound (env/train ratio 11.60)
- cpu: training-bound (env/train ratio 7.62)
