# Benchmark Summary

- run_tag: 20260120_101512_rerun
- timestamp: 1768933130.367281
- env_id: CartPole-v1
- config_path: configs/cartpole.json
- config_summary: env_id=CartPole-v1, seed=42, gamma=0.99, replay_capacity=100000, batch_size=32, total_frames=500000
- frames: 5000
- mode: both

## Results

| device | env_fps | train_fps |
| --- | --- | --- |
| mps | 128777.06 | 88.28 |
| cpu | 133156.09 | 640.25 |

## Recommendation

Best training throughput: cpu (640.25 fps).

## Bottleneck hints

Heuristic based on env_fps vs train_fps (higher ratio means training dominates).
- mps: training-bound (env/train ratio 1458.72)
- cpu: training-bound (env/train ratio 207.97)
