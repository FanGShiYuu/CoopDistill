"""Evaluate the complete CoopDistill checkpoint on the default benchmark."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import torch

from .benchmark import load_default_benchmark
from .trainer import Actor, atomic_json, evaluate, initialize_worker, mean_metrics


ROOT = Path(__file__).resolve().parents[1]


def load_actor(path: Path) -> Actor:
    actor = Actor()
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0
        state = torch.load(path, map_location="cpu")
    actor.load_state_dict(state)
    actor.eval()
    return actor


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
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "results/reproduced.json")
    args = parser.parse_args()

    cfg, manifest, cases = load_default_benchmark(args.benchmark)
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        parser.error("The benchmark contains no cases")

    actor = load_actor(args.checkpoint)
    executor = None
    if args.workers:
        executor = ProcessPoolExecutor(args.workers, initializer=initialize_worker)

    started = time.monotonic()
    try:
        print(f"EVALUATE model=coopdistill cases={len(cases)}", flush=True)
        episodes = evaluate(cases, cfg, "full", actor, executor)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    rows = [episode["summary"] for episode in episodes]
    metrics = mean_metrics(rows)
    payload = {
        "status": "complete",
        "model": "CoopDistill",
        "benchmark_cases": len(cases),
        "benchmark_selected_cases": int(manifest["selected_count"]),
        "checkpoint": str(args.checkpoint),
        "runtime_s": time.monotonic() - started,
        "metrics": metrics,
        "episodes": rows,
    }
    atomic_json(args.output, payload)
    print(
        json.dumps(
            {
                "success_percent": 100.0 * metrics["success"],
                "collision_percent": 100.0 * metrics["collision"],
                "cleared_percent": 100.0 * metrics["cleared_fraction"],
                "delay_s": metrics["delay_censored_s"],
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
