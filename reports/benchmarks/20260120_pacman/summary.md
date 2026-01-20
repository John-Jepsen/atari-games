# Benchmark Summary

- run_tag: 20260120_pacman
- timestamp: 1768935072.492779
- env_id: ALE/MsPacman-v5
- config_path: configs/pacman.json
- config_summary: env_id=ALE/MsPacman-v5, seed=42, gamma=0.99, replay_capacity=1000000, batch_size=32, total_frames=10000000
- frames: 5000
- mode: both

## Results

| device | env_fps | train_fps |
| --- | --- | --- |
| mps | 1644.95 | 138.29 |
| cpu | 1612.58 | 245.32 |

## Recommendation

Best training throughput: cpu (245.32 fps).

## Bottleneck hints

Heuristic based on env_fps vs train_fps (higher ratio means training dominates).
- mps: training-bound (env/train ratio 11.90)
- cpu: training-bound (env/train ratio 6.57)
