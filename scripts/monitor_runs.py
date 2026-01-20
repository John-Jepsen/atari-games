#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


@dataclass
class MetricSnapshot:
    env_name: str
    episodes: int
    last_reward: float
    last_epsilon: float
    last_loss: float
    avg_reward: float
    path: Path
    updated_seconds_ago: int


@dataclass
class SystemSnapshot:
    load1: float
    load5: float
    load15: float
    mem_used_mb: float
    mem_free_mb: float


def _run_ps() -> List[str]:
    try:
        output = subprocess.check_output(["ps", "-ax", "-o", "pid=,command="]).decode()
    except Exception:
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def find_training_pids() -> List[Tuple[str, int]]:
    pids: List[Tuple[str, int]] = []
    for line in _run_ps():
        if "train/train_cartpole.py" in line or "train/train_atari.py" in line:
            try:
                pid_str, cmd = line.split(" ", 1)
                pids.append((cmd, int(pid_str)))
            except ValueError:
                continue
    return pids


def _tail_lines(path: Path, count: int) -> List[str]:
    if count <= 0:
        return []
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = 4096
            data = b""
            while size > 0 and data.count(b"\n") <= count:
                read_size = min(block, size)
                f.seek(size - read_size)
                data = f.read(read_size) + data
                size -= read_size
        lines = data.splitlines()[-count:]
        return [line.decode() for line in lines]
    except Exception:
        return []


def _parse_metrics(path: Path, tail: int) -> Optional[MetricSnapshot]:
    lines = _tail_lines(path, tail + 1)
    if len(lines) <= 1:
        return None
    records = []
    for line in lines:
        if line.startswith("frame,"):
            continue
        parts = line.split(",")
        if len(parts) < 5:
            continue
        try:
            frame = int(parts[0])
            episode = int(parts[1])
            reward = float(parts[2])
            epsilon = float(parts[3])
            loss = float(parts[4])
            records.append((frame, episode, reward, epsilon, loss))
        except ValueError:
            continue
    if not records:
        return None

    last = records[-1]
    avg_reward = sum(r[2] for r in records) / float(len(records))
    env_name = path.name.replace("metrics_", "").replace(".csv", "")
    updated_seconds_ago = int(time.time() - path.stat().st_mtime)
    return MetricSnapshot(
        env_name=env_name,
        episodes=last[1],
        last_reward=last[2],
        last_epsilon=last[3],
        last_loss=last[4],
        avg_reward=avg_reward,
        path=path,
        updated_seconds_ago=updated_seconds_ago,
    )


def collect_metrics(reports_dir: Path, tail: int) -> List[MetricSnapshot]:
    snapshots = []
    for path in sorted(reports_dir.glob("metrics_*.csv")):
        snapshot = _parse_metrics(path, tail)
        if snapshot:
            snapshots.append(snapshot)
    return snapshots


def _get_loadavg() -> Tuple[float, float, float]:
    try:
        return os.getloadavg()
    except Exception:
        return (0.0, 0.0, 0.0)


def _get_memory_mb() -> Tuple[float, float]:
    if sys.platform == "darwin":
        try:
            output = subprocess.check_output(["vm_stat"]).decode()
            lines = output.splitlines()
            page_size = 4096
            for line in lines:
                if "page size of" in line:
                    try:
                        page_size = int(line.split("page size of")[1].split("bytes")[0].strip())
                    except Exception:
                        page_size = 4096
            stats = {}
            for line in lines:
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                value = value.strip().strip(".")
                try:
                    stats[key] = int(value)
                except ValueError:
                    continue
            free_pages = stats.get("Pages free", 0) + stats.get("Pages speculative", 0)
            used_pages = (
                stats.get("Pages active", 0)
                + stats.get("Pages inactive", 0)
                + stats.get("Pages wired down", 0)
                + stats.get("Pages compressed", 0)
            )
            mem_free_mb = (free_pages * page_size) / (1024 ** 2)
            mem_used_mb = (used_pages * page_size) / (1024 ** 2)
            return (mem_used_mb, mem_free_mb)
        except Exception:
            return (0.0, 0.0)

    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        total_kb = 0
        avail_kb = 0
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal:"):
                total_kb = int(line.split()[1])
            elif line.startswith("MemAvailable:"):
                avail_kb = int(line.split()[1])
        mem_used_mb = (total_kb - avail_kb) / 1024.0
        mem_free_mb = avail_kb / 1024.0
        return (mem_used_mb, mem_free_mb)

    return (0.0, 0.0)


def collect_system_snapshot() -> SystemSnapshot:
    load1, load5, load15 = _get_loadavg()
    mem_used_mb, mem_free_mb = _get_memory_mb()
    return SystemSnapshot(
        load1=load1,
        load5=load5,
        load15=load15,
        mem_used_mb=mem_used_mb,
        mem_free_mb=mem_free_mb,
    )


def _notify(message: str, title: str = "Atari Training") -> None:
    try:
        subprocess.run(
            [
                "osascript",
                "-e",
                f'display notification "{message}" with title "{title}"',
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return


def _read_last_event(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = 4096
            data = b""
            while size > 0 and data.count(b"\n") <= 1:
                read_size = min(block, size)
                f.seek(size - read_size)
                data = f.read(read_size) + data
                size -= read_size
        lines = [line for line in data.splitlines() if line.strip()]
        if not lines:
            return None
        return json.loads(lines[-1].decode())
    except Exception:
        return None


def _active_envs(pids: List[Tuple[str, int]]) -> set[str]:
    envs = set()
    for cmd, _ in pids:
        if "train/train_cartpole.py" in cmd:
            envs.add("CartPole-v1")
        if "configs/space_invaders" in cmd:
            envs.add("ALE_SpaceInvaders-v5")
        if "configs/pacman" in cmd:
            envs.add("ALE_MsPacman-v5")
        if "train/train_atari.py" in cmd and "configs/" in cmd:
            # Best-effort parse for custom config names
            parts = cmd.split()
            if "--config" in parts:
                idx = parts.index("--config")
                if idx + 1 < len(parts):
                    cfg = Path(parts[idx + 1]).name
                    if "space" in cfg and "invader" in cfg:
                        envs.add("ALE_SpaceInvaders-v5")
                    if "pacman" in cfg:
                        envs.add("ALE_MsPacman-v5")
    return envs


def print_status(
    pids: List[Tuple[str, int]],
    snapshots: List[MetricSnapshot],
    system: SystemSnapshot,
    events_dir: Path,
    notify: bool,
    stale_seconds: int,
    min_avg_reward_delta: float,
    prev_alerts: dict,
) -> None:
    print("\n=== Training Processes ===")
    if not pids:
        print("No training processes found.")
    else:
        for cmd, pid in pids:
            print(f"PID {pid}: {cmd}")

    print("\n=== System ===")
    print(
        "load(avg): "
        f"{system.load1:.2f} {system.load5:.2f} {system.load15:.2f} | "
        f"mem_used={system.mem_used_mb:.0f}MB mem_free={system.mem_free_mb:.0f}MB"
    )

    print("\n=== Metrics (last episodes) ===")
    if not snapshots:
        print("No metrics found.")
        return
    active_envs = _active_envs(pids)
    for snap in snapshots:
        print(
            f"{snap.env_name}: episodes={snap.episodes} "
            f"last_reward={snap.last_reward:.2f} avg_reward={snap.avg_reward:.2f} "
            f"epsilon={snap.last_epsilon:.3f} loss={snap.last_loss:.4f} "
            f"updated={snap.updated_seconds_ago}s ago"
        )
        alert_key = f"{snap.env_name}-stale"
        if snap.env_name in active_envs and stale_seconds and snap.updated_seconds_ago > stale_seconds:
            msg = f"{snap.env_name} stalled (> {stale_seconds}s without updates)."
            print(f"ALERT: {msg}")
            if notify and not prev_alerts.get(alert_key):
                _notify(msg)
                prev_alerts[alert_key] = True
        else:
            prev_alerts.pop(alert_key, None)

        event_path = events_dir / f"events_{snap.env_name}.jsonl"
        last_event = _read_last_event(event_path)
        if last_event:
            print(f"  last_event: {last_event.get('type')} @ frame {last_event.get('frame')}")

    if min_avg_reward_delta and snapshots:
        for snap in snapshots:
            if snap.env_name not in active_envs:
                continue
            key = f"{snap.env_name}-avg"
            prev = prev_alerts.get(key)
            if prev is None:
                prev_alerts[key] = snap.avg_reward
                continue
            if (snap.avg_reward - prev) < min_avg_reward_delta and notify:
                _notify(f"{snap.env_name} avg reward flat (< {min_avg_reward_delta} gain).")
            prev_alerts[key] = snap.avg_reward


def _write_snapshot_header(path: Path) -> None:
    path.write_text(
        "timestamp,env_name,episodes,last_reward,avg_reward,epsilon,loss,updated_seconds_ago,"
        "load1,load5,load15,mem_used_mb,mem_free_mb,pids\n"
    )


def append_snapshots(
    path: Path,
    system: SystemSnapshot,
    snapshots: List[MetricSnapshot],
    pids: List[Tuple[str, int]],
) -> None:
    if not path.exists():
        _write_snapshot_header(path)
    pid_list = "|".join(str(pid) for _, pid in pids)
    ts = int(time.time())
    with path.open("a") as f:
        for snap in snapshots:
            f.write(
                f"{ts},{snap.env_name},{snap.episodes},{snap.last_reward:.3f},{snap.avg_reward:.3f},"
                f"{snap.last_epsilon:.4f},{snap.last_loss:.6f},{snap.updated_seconds_ago},"
                f"{system.load1:.3f},{system.load5:.3f},{system.load15:.3f},"
                f"{system.mem_used_mb:.1f},{system.mem_free_mb:.1f},{pid_list}\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor long-running DQN training runs.")
    parser.add_argument("--reports", default="reports", help="Reports directory path.")
    parser.add_argument("--tail", type=int, default=20, help="Episodes to average over.")
    parser.add_argument("--interval", type=int, default=30, help="Refresh interval in seconds.")
    parser.add_argument("--once", action="store_true", help="Print once and exit.")
    parser.add_argument("--notify", action="store_true", help="Send macOS notifications on alerts.")
    parser.add_argument("--stale-seconds", type=int, default=300, help="Alert if no updates for N seconds.")
    parser.add_argument(
        "--min-avg-reward-delta",
        type=float,
        default=0.0,
        help="Alert if avg reward doesn't improve by this delta.",
    )
    parser.add_argument("--events", default="reports", help="Directory for events_*.jsonl logs.")
    parser.add_argument(
        "--snapshot-out",
        default="reports/monitor_snapshots.csv",
        help="CSV output path for periodic snapshots.",
    )
    args = parser.parse_args()

    reports_dir = Path(args.reports)
    snapshot_path = Path(args.snapshot_out)
    events_dir = Path(args.events)
    prev_alerts: dict = {}

    while True:
        pids = find_training_pids()
        system = collect_system_snapshot()
        snapshots = collect_metrics(reports_dir, args.tail)
        print_status(
            pids,
            snapshots,
            system,
            events_dir,
            args.notify,
            args.stale_seconds,
            args.min_avg_reward_delta,
            prev_alerts,
        )
        if snapshots:
            append_snapshots(snapshot_path, system, snapshots, pids)
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
