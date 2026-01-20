# Monitoring

Run a lightweight monitor to track training progress and recent rewards:

```
. .venv/bin/activate
python scripts/monitor_runs.py --interval 30
```

One-shot snapshot:

```
. .venv/bin/activate
python scripts/monitor_runs.py --once
```

## Benchmark throughput

Run a short benchmark to compare environment step speed and training speed across devices:

```
. .venv/bin/activate
python scripts/benchmark_training.py --config configs/cartpole.json --device mps --frames 20000
python scripts/benchmark_training.py --config configs/cartpole.json --device cpu --frames 20000
```

Compare devices in one command:

```
. .venv/bin/activate
python scripts/compare_benchmarks.py --config configs/cartpole.json --frames 20000
```

This also writes a Markdown summary to `reports/benchmarks/<tag>/summary.md`.

CSV snapshots are appended to `reports/monitor_snapshots.csv` by default. You can
change the output path:

```
. .venv/bin/activate
python scripts/monitor_runs.py --snapshot-out reports/my_snapshots.csv
```
