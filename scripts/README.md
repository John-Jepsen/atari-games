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
