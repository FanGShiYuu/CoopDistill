"""Evaluate CoopDistill and independent baselines on the default benchmark."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import torch

from .comparison import STATIC_METHODS, load_hard_protocol, static_rollout
from .trainer import Actor, atomic_json, evaluate, initialize_worker, mean_metrics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METHODS = (
    "pg",
    "coopdistill",
    "fcfs_reservation",
    "auction_reservation",
    "nash_bargaining_mpc",
)


def load_actor(path: Path) -> Actor:
    actor = Actor()
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0
        state = torch.load(path, map_location="cpu")
    actor.load_state_dict(state)
    actor.eval()
    return actor


def display_name(method: str) -> str:
    return {
        "pg": "Potential Game",
        "coopdistill": "CoopDistill",
        "fcfs_reservation": "FCFS reservation",
        "auction_reservation": "Auction reservation",
        "nash_bargaining_mpc": "Nash-bargaining MPC",
    }.get(method, method)


def table_row(method: str, metrics: dict[str, float]) -> dict[str, float | str]:
    return {
        "method": display_name(method),
        "success_percent": 100.0 * metrics["success"],
        "collision_percent": 100.0 * metrics["collision"],
        "cleared_percent": 100.0 * metrics["cleared_fraction"],
        "delay_s": metrics["delay_censored_s"],
    }


def print_table(rows: list[dict[str, float | str]]) -> None:
    print("| Method | Success (%) | Collision (%) | Cleared (%) | Delay (s) |")
    print("|---|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['method']} | {row['success_percent']:.1f} | "
            f"{row['collision_percent']:.1f} | {row['cleared_percent']:.1f} | "
            f"{row['delay_s']:.1f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "benchmarks/default",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints/coopdistill_seed23.pt",
    )
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--mpc-horizon", type=int, default=4)
    parser.add_argument("--shooting-candidates", type=int, default=20)
    parser.add_argument("--output", type=Path, default=ROOT / "results/reproduced.json")
    args = parser.parse_args()

    methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    allowed = set(DEFAULT_METHODS) | set(STATIC_METHODS)
    unknown = set(methods) - allowed
    if not methods or unknown:
        parser.error(f"Unknown methods: {sorted(unknown)}")

    cfg, manifest, cases = load_hard_protocol(args.benchmark)
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        parser.error("The benchmark contains no cases")

    actor = load_actor(args.checkpoint) if "coopdistill" in methods else None
    executor = None
    if args.workers:
        executor = ProcessPoolExecutor(args.workers, initializer=initialize_worker)

    started = time.monotonic()
    conditions: dict[str, list[dict[str, float]]] = {}
    try:
        if "pg" in methods:
            tick = time.monotonic()
            print(f"EVALUATE method=pg cases={len(cases)}", flush=True)
            episodes = evaluate(cases, cfg, "pg", None, executor)
            conditions["pg"] = [episode["summary"] for episode in episodes]
            print(f"DONE method=pg seconds={time.monotonic() - tick:.1f}", flush=True)
        if "coopdistill" in methods:
            tick = time.monotonic()
            print(f"EVALUATE method=coopdistill cases={len(cases)}", flush=True)
            episodes = evaluate(cases, cfg, "full", actor, executor)
            conditions["coopdistill"] = [episode["summary"] for episode in episodes]
            print(f"DONE method=coopdistill seconds={time.monotonic() - tick:.1f}", flush=True)

        for method in methods:
            if method not in STATIC_METHODS:
                continue
            tick = time.monotonic()
            print(f"EVALUATE method={method} cases={len(cases)}", flush=True)
            tasks = [
                (method, case, cfg, args.mpc_horizon, args.shooting_candidates)
                for case in cases
            ]
            if executor is None:
                conditions[method] = [static_rollout(task) for task in tasks]
            else:
                conditions[method] = list(executor.map(static_rollout, tasks, chunksize=1))
            print(f"DONE method={method} seconds={time.monotonic() - tick:.1f}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    aggregate = {method: mean_metrics(rows) for method, rows in conditions.items()}
    ordered = [method for method in methods if method in aggregate]
    table = [table_row(method, aggregate[method]) for method in ordered]
    payload = {
        "status": "complete",
        "benchmark_cases": len(cases),
        "benchmark_selected_cases": int(manifest["selected_count"]),
        "checkpoint": str(args.checkpoint),
        "runtime_s": time.monotonic() - started,
        "aggregate": aggregate,
        "table": table,
    }
    atomic_json(args.output, payload)
    print_table(table)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
