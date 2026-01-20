# Reports

Place the project blog post and result figures here.
Suggested outline:
- Problem statement and goals
- DQN baseline and key design choices
- Training setup and hyperparameters
- Results (learning curves, scores)
- Limitations and next steps

## Benchmark summaries

Benchmarks compare environment throughput and training throughput across devices
(e.g., CPU vs MPS) using the same DQN training loop. Each run generates a Markdown
summary at:

```
reports/benchmarks/<tag>/summary.md
```

Example commands:

```
. .venv/bin/activate
python scripts/compare_benchmarks.py --config configs/cartpole.json --frames 5000
python scripts/compare_benchmarks.py --config configs/space_invaders.json --frames 5000 --tag 20260120_space_invaders
python scripts/compare_benchmarks.py --config configs/pacman.json --frames 5000 --tag 20260120_pacman
```
