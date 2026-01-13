from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config not found: {config_path}")
    try:
        data = json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {config_path}: {exc}") from exc
    return data


def require_keys(data: dict[str, Any], keys: list[str], ctx: str) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise ConfigError(f"Missing {ctx} keys: {', '.join(missing)}")


def summarize_config(data: dict[str, Any]) -> str:
    env_id = data.get("env_id", "<unset>")
    seed = data.get("seed", "<unset>")
    dqn = data.get("dqn", {})
    training = data.get("training", {})
    return (
        f"env_id={env_id}, seed={seed}, "
        f"gamma={dqn.get('gamma')}, replay_capacity={dqn.get('replay_capacity')}, "
        f"batch_size={dqn.get('batch_size')}, total_frames={training.get('total_frames')}"
    )
