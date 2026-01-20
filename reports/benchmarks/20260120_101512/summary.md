# Benchmark Summary

- run_tag: 20260120_101512
- timestamp: 1768932977.9417758
- env_id: CartPole-v1
- config_path: configs/cartpole.json
- config_summary: env_id=CartPole-v1, seed=42, gamma=0.99, replay_capacity=100000, batch_size=32, total_frames=500000
- frames: 5000
- mode: both

## Results

| device | env_fps | train_fps |
| --- | --- | --- |
| mps | 129183.95 | 79.57 |
| cpu | 130760.55 | 637.49 |

## Recommendation

Best training throughput: cpu (637.49 fps).

## Bottleneck hints

Heuristic based on env_fps vs train_fps (lower ratio means env likely dominates).
- mps: training-bound (env/train ratio 1623.44)
- cpu: training-bound (env/train ratio 205.12)
