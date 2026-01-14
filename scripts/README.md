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

CSV snapshots are appended to `reports/monitor_snapshots.csv` by default. You can
change the output path:

```
. .venv/bin/activate
python scripts/monitor_runs.py --snapshot-out reports/my_snapshots.csv
```
