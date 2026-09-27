"""Same-protocol baselines for the random-OD intersection benchmark.

No comparison method calls the potential-game solver. Static controllers and
discrete MADQN share the frozen environment, cases, horizon, and metrics used
by the PG-residual MAPPO checkpoint.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from torch import nn

from .environment import Config, IntersectionEnv
from .hard_cases import case_from_dict
from .scenarios import MAX_VEHICLES, make_case, suite
from .trainer import atomic_json, initialize_worker, mean_metrics, validation_rank


STATIC_METHODS = (
    "fcfs_reservation",
    "auction_reservation",
    "priority_mpc",
    "nash_bargaining_mpc",
)
STOP = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_hard_protocol(root: Path):
    protocol = read(root / "protocol.json")
    manifest = read(root / "manifest.json")
    if manifest["protocol_sha256"] != sha256(root / "protocol.json"):
        raise ValueError("Hard-case protocol/manifest mismatch")
    cases = [case_from_dict(item["case"]) for item in manifest["cases"]]
    if [case.fingerprint for case in cases] != [item["pg"]["fingerprint"] for item in manifest["cases"]]:
        raise ValueError("Hard-case physical definitions changed")
    return Config(**protocol["environment"]), manifest, cases


def conflict_distance(env: IntersectionEnv, vehicle: int) -> float:
    distances = []
    for pair in env.conflicts:
        if pair.i == vehicle:
            distances.append(pair.entry_i - env.progress[vehicle])
        elif pair.j == vehicle:
            distances.append(pair.entry_j - env.progress[vehicle])
    ahead = [distance for distance in distances if distance > -env.cfg.vehicle_length]
    return min(ahead, default=math.inf)


def eta(env: IntersectionEnv, vehicle: int) -> float:
    distance = conflict_distance(env, vehicle)
    return distance / max(env.speed[vehicle], 0.2) if math.isfinite(distance) else math.inf


def stop_command(env: IntersectionEnv, vehicle: int, entry_distance: float) -> float:
    distance = max(0.0, entry_distance - env.cfg.stop_gap)
    if distance <= 0.05:
        return -env.cfg.max_brake
    target_speed = math.sqrt(max(0.0, 2.0 * env.cfg.max_brake * distance))
    return min(0.0, (target_speed - env.speed[vehicle]) / env.cfg.pg_dt)


def reservation_action(env: IntersectionEnv, priority: np.ndarray) -> np.ndarray:
    """Apply an acyclic all-vehicle reservation order to CAV accelerations."""
    action = env.nominal_accelerations()
    for pair in env.conflicts:
        i, j = pair.i, pair.j
        if env.finished[i] or env.finished[j]:
            continue
        di, dj = pair.entry_i - env.progress[i], pair.entry_j - env.progress[j]
        xi, xj = pair.exit_i - env.progress[i], pair.exit_j - env.progress[j]
        if min(xi, xj) < -env.cfg.vehicle_length:
            continue
        winner, loser, loser_distance = (i, j, dj) if priority[i] < priority[j] else (j, i, di)
        winner_exit = xi if winner == i else xj
        if loser_distance <= 30.0 and winner_exit >= -env.cfg.vehicle_length and env.cav[loser]:
            action[loser] = min(action[loser], stop_command(env, loser, loser_distance))
    return env.project_physical(action)


class FCFSController:
    def __init__(self, env: IntersectionEnv):
        arrival = np.asarray([eta(env, i) for i in range(env.n)])
        order = np.lexsort((np.arange(env.n), arrival))
        self.priority = np.empty(env.n, dtype=int)
        self.priority[order] = np.arange(env.n)

    def action(self, env: IntersectionEnv) -> np.ndarray:
        return reservation_action(env, self.priority)


class AuctionController:
    def action(self, env: IntersectionEnv) -> np.ndarray:
        arrivals = np.asarray([eta(env, i) for i in range(env.n)])
        proximity = np.asarray([
            0.0 if not math.isfinite(conflict_distance(env, i))
            else 1.0 / (max(conflict_distance(env, i), 0.0) + 2.0)
            for i in range(env.n)
        ])
        bids = 0.10 * env.wait_time + 3.0 * proximity + 1.0 / (arrivals + 0.5)
        order = np.lexsort((np.arange(env.n), -bids))
        priority = np.empty(env.n, dtype=int)
        priority[order] = np.arange(env.n)
        return reservation_action(env, priority)


def raw_from_absolute(acceleration: np.ndarray, lateral: np.ndarray) -> np.ndarray:
    raw = np.zeros((MAX_VEHICLES, 3), dtype=np.float32)
    raw[: len(acceleration), 0] = np.arctanh(np.clip((acceleration + 1.0) / 3.0, -0.995, 0.995))
    raw[: len(lateral), 2] = np.arctanh(np.clip(lateral, -0.995, 0.995))
    return raw


def candidate_prediction(env: IntersectionEnv, acceleration: np.ndarray, lateral: np.ndarray, horizon: int):
    raw = raw_from_absolute(acceleration, lateral)
    prediction, executable = env.proposal(raw, lateral=True, absolute=True)
    initial_progress = prediction.progress.copy()
    for _ in range(horizon):
        if prediction.done:
            break
        prediction.advance(executable)
    return prediction, prediction.progress - initial_progress


def priority_candidates(env: IntersectionEnv):
    active = np.flatnonzero(env.cav & ~env.finished)
    arrivals = np.asarray([eta(env, i) for i in range(env.n)])
    orders = [np.argsort(arrivals), np.argsort(-arrivals)]
    orders.extend(np.asarray([winner] + [i for i in range(env.n) if i != winner]) for winner in active)
    yielded = set()
    for order in orders:
        key = tuple(int(i) for i in order)
        if key in yielded:
            continue
        yielded.add(key)
        rank = np.empty(env.n, dtype=int)
        rank[order] = np.arange(env.n)
        acceleration = reservation_action(env, rank)
        for winner in active:
            if rank[winner] == 0:
                acceleration[winner] = min(env.cfg.max_acceleration, acceleration[winner] + 0.8)
        yield acceleration, np.zeros(env.n)


class PriorityMPCController:
    def __init__(self, horizon: int = 4):
        self.horizon = horizon

    def action(self, env: IntersectionEnv):
        best_rank, best = None, None
        for acceleration, lateral in priority_candidates(env):
            prediction, progress = candidate_prediction(env, acceleration, lateral, self.horizon)
            rank = (
                int(prediction.cav_collision), int(prediction.collision), int(prediction.boundary),
                int(prediction.dynamic_violation), -int(prediction.finished.sum()),
                -float(progress.sum()), float(np.mean(np.abs(acceleration - env.acceleration))),
            )
            if best_rank is None or rank < best_rank:
                best_rank, best = rank, (acceleration, lateral)
        return best


def shooting_candidates(env: IntersectionEnv, count: int = 20):
    nominal = env.nominal_accelerations()
    active = np.flatnonzero(env.cav & ~env.finished)
    yield nominal.copy(), np.zeros(env.n)
    yield from priority_candidates(env)
    seed = int(env.case.seed + round(env.time / env.cfg.pg_dt) * 7919)
    rng = np.random.default_rng(seed)
    for _ in range(count):
        acceleration = nominal.copy()
        acceleration[active] = rng.choice((-env.cfg.max_brake, -1.0, 0.5, env.cfg.max_acceleration), len(active))
        lateral = np.zeros(env.n)
        lateral[active] = rng.choice((-0.75, 0.0, 0.75), len(active))
        yield acceleration, lateral


class NashBargainingMPCController:
    def __init__(self, horizon: int = 4, candidates: int = 20):
        self.horizon, self.candidates = horizon, candidates

    def action(self, env: IntersectionEnv):
        active = np.flatnonzero(env.cav & ~env.finished)
        best_rank, best = None, None
        for acceleration, lateral in shooting_candidates(env, self.candidates):
            prediction, progress = candidate_prediction(env, acceleration, lateral, self.horizon)
            utilities = np.maximum(progress[active], 0.05)
            nash = float(np.log(utilities).sum()) if len(utilities) else 0.0
            fairness = float(np.std(utilities)) if len(utilities) else 0.0
            rank = (
                int(prediction.cav_collision), int(prediction.collision), int(prediction.boundary),
                int(prediction.dynamic_violation), -int(prediction.finished.sum()),
                -nash, fairness, -float(progress.sum()),
            )
            if best_rank is None or rank < best_rank:
                best_rank, best = rank, (acceleration, lateral)
        return best


def static_rollout(task):
    method, case, cfg, horizon, shooting = task
    env = IntersectionEnv(case, cfg)
    controller = {
        "fcfs_reservation": lambda: FCFSController(env),
        "auction_reservation": AuctionController,
        "priority_mpc": lambda: PriorityMPCController(horizon),
        "nash_bargaining_mpc": lambda: NashBargainingMPCController(horizon, shooting),
    }[method]()
    while not env.done:
        output = controller.action(env)
        if isinstance(output, tuple):
            acceleration, lateral = output
            env.step(raw_from_absolute(acceleration, lateral), mode="direct", lateral=True)
        else:
            env.advance(output)
    return env.summary()


class QNetwork(nn.Module):
    def __init__(self, actions: int = 9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(44, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, actions)
        )

    def forward(self, observation):
        return self.net(observation)


ACTION_LIBRARY = np.asarray(
    [(acceleration, lateral) for acceleration in (-4.0, -1.0, 2.0) for lateral in (-0.75, 0.0, 0.75)],
    dtype=np.float32,
)


def madqn_raw(action_ids: np.ndarray, n: int) -> np.ndarray:
    selected = ACTION_LIBRARY[np.asarray(action_ids, dtype=int)]
    return raw_from_absolute(selected[:n, 0], selected[:n, 1])


def madqn_episode(case, cfg, qnet, device, epsilon, rng, replay=None):
    env = IntersectionEnv(case, cfg)
    total = 0.0
    while not env.done:
        observation = env.observe(include_pg=False)
        mask = env.mask()
        with torch.no_grad():
            greedy = qnet(torch.as_tensor(observation, dtype=torch.float32, device=device)).argmax(-1).cpu().numpy()
        actions = greedy.copy()
        explore = rng.random(MAX_VEHICLES) < epsilon
        actions[explore] = rng.integers(0, len(ACTION_LIBRARY), size=int(explore.sum()))
        reward, _ = env.step(madqn_raw(actions, env.n), mode="direct", lateral=True)
        next_observation = env.observe(include_pg=False)
        next_mask = env.mask()
        if replay is not None:
            for agent in np.flatnonzero(mask):
                replay.append((observation[agent].copy(), int(actions[agent]), float(reward),
                               next_observation[agent].copy(), float(not next_mask[agent])))
        total += reward
    return total, env.summary()


def optimize_madqn(qnet, target, optimizer, replay, batch_size, gamma, device, rng):
    indices = rng.choice(len(replay), size=batch_size, replace=False)
    batch = [replay[int(index)] for index in indices]
    obs = torch.as_tensor(np.asarray([row[0] for row in batch]), dtype=torch.float32, device=device)
    action = torch.as_tensor([row[1] for row in batch], dtype=torch.long, device=device)
    reward = torch.as_tensor([row[2] for row in batch], dtype=torch.float32, device=device)
    nxt = torch.as_tensor(np.asarray([row[3] for row in batch]), dtype=torch.float32, device=device)
    done = torch.as_tensor([row[4] for row in batch], dtype=torch.float32, device=device)
    q = qnet(obs).gather(1, action[:, None]).squeeze(1)
    with torch.no_grad():
        next_action = qnet(nxt).argmax(1)
        expected = reward + gamma * (1.0 - done) * target(nxt).gather(1, next_action[:, None]).squeeze(1)
    loss = nn.functional.smooth_l1_loss(q, expected)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(qnet.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach().cpu())


def evaluate_madqn(cases, cfg, qnet, device):
    qnet.eval()
    rng = np.random.default_rng(0)
    rows = [madqn_episode(case, cfg, qnet, device, 0.0, rng)[1] for case in cases]
    qnet.train()
    return rows


def run_static(args):
    root, output = Path(args.hard_root), Path(args.output)
    cfg, _, cases = load_hard_protocol(root)
    methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    unknown = set(methods) - set(STATIC_METHODS)
    if not methods or unknown:
        raise ValueError(f"Invalid static methods: {sorted(unknown)}")
    if args.limit:
        cases = cases[:args.limit]
    tasks = [(method, case, cfg, args.mpc_horizon, args.shooting_candidates)
             for method in methods for case in cases]
    started = time.monotonic()
    with ProcessPoolExecutor(args.workers, initializer=initialize_worker) as executor:
        results = list(executor.map(static_rollout, tasks, chunksize=1))
    conditions, offset = {}, 0
    for method in methods:
        conditions[method] = results[offset:offset + len(cases)]
        offset += len(cases)
    payload = {
        "status": "complete", "experiment_type": "same_protocol_static_comparison",
        "manifest_sha256": sha256(root / "manifest.json"), "unique_test_cases": len(cases),
        "runtime_s": time.monotonic() - started, "methods": list(methods), "conditions": conditions,
        "aggregate": {method: mean_metrics(rows) for method, rows in conditions.items()},
    }
    atomic_json(output, payload)
    print("STATIC_COMPLETE " + json.dumps(payload["aggregate"]), flush=True)


def run_madqn(args):
    root, output = Path(args.hard_root), Path(args.output)
    cfg, _, hard_cases = load_hard_protocol(root)
    if args.limit:
        hard_cases = hard_cases[:args.limit]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("MADQN requested CUDA but CUDA is unavailable")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    qnet, target = QNetwork().to(device), QNetwork().to(device)
    target.load_state_dict(qnet.state_dict())
    optimizer = torch.optim.Adam(qnet.parameters(), lr=args.lr)
    replay = deque(maxlen=args.replay_size)
    validation = suite("validation", 1)
    if args.limit:
        validation = validation[:args.limit]
    best_rank, best_state, best_episode = None, None, -1
    train_base = 4_000_000 + args.seed * 100_000
    started = time.monotonic()
    start_episode = 0
    resume_path = output.with_suffix(".last.pt")
    if args.resume and resume_path.exists():
        state = torch.load(resume_path, map_location=device, weights_only=False)
        qnet.load_state_dict(state["qnet"])
        target.load_state_dict(state["target"])
        optimizer.load_state_dict(state["optimizer"])
        replay.extend(state.get("replay", ()))
        start_episode = int(state["episode"])
        best_rank = state.get("best_rank")
        best_state = state.get("best_state")
        best_episode = int(state.get("best_episode", -1))
        if "rng_state" in state:
            rng.bit_generator.state = state["rng_state"]
        print(f"MADQN_RESUME episode={start_episode}/{args.episodes} replay={len(replay)}", flush=True)
    global STOP

    def stop(*_):
        global STOP
        STOP = True

    signal.signal(signal.SIGTERM, stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, stop)
    for episode in range(start_episode, args.episodes):
        fraction = min(1.0, episode / max(args.epsilon_decay, 1))
        epsilon = args.epsilon_start + fraction * (args.epsilon_end - args.epsilon_start)
        total, _ = madqn_episode(make_case(train_base + episode), cfg, qnet, device, epsilon, rng, replay)
        losses = []
        if len(replay) >= args.warmup:
            for _ in range(args.gradient_steps):
                losses.append(optimize_madqn(qnet, target, optimizer, replay, args.batch_size,
                                             args.gamma, device, rng))
        if (episode + 1) % args.target_update == 0:
            target.load_state_dict(qnet.state_dict())
        if (episode + 1) % args.validation_every == 0 or episode == 0 or episode + 1 == args.episodes:
            metrics = mean_metrics(evaluate_madqn(validation, cfg, qnet, device))
            rank = validation_rank(metrics)
            if best_rank is None or rank > best_rank:
                best_rank, best_episode = rank, episode + 1
                best_state = {key: value.detach().cpu().clone() for key, value in qnet.state_dict().items()}
            print(f"MADQN episode={episode + 1}/{args.episodes} return={total:.3f} "
                  f"epsilon={epsilon:.3f} loss={np.mean(losses) if losses else 0.:.5f} "
                  f"validation={json.dumps(metrics)}", flush=True)
        if STOP:
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"qnet": qnet.state_dict(), "target": target.state_dict(),
                        "optimizer": optimizer.state_dict(), "episode": episode + 1,
                        "replay": list(replay), "best_rank": best_rank,
                        "best_state": best_state, "best_episode": best_episode,
                        "rng_state": rng.bit_generator.state},
                       output.with_suffix(".last.pt"))
            raise SystemExit(75)
    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in qnet.state_dict().items()}
    qnet.load_state_dict(best_state)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output.with_suffix(".pt"))
    rows = evaluate_madqn(hard_cases, cfg, qnet, device)
    payload = {
        "status": "complete", "experiment_type": "independent_discrete_madqn",
        "training_seed": args.seed, "best_episode": best_episode,
        "manifest_sha256": sha256(root / "manifest.json"), "unique_test_cases": len(hard_cases),
        "runtime_s": time.monotonic() - started, "action_library": ACTION_LIBRARY.tolist(),
        "policy": rows, "aggregate": mean_metrics(rows),
    }
    atomic_json(output, payload)
    print("MADQN_COMPLETE " + json.dumps(payload["aggregate"]), flush=True)


def validate_frozen_result(path, manifest_hash, variant, seed):
    data = read(path)
    if (data.get("status") != "complete" or data.get("variant") != variant
            or data.get("training_seed") != seed or data.get("manifest_sha256") != manifest_hash):
        raise ValueError(f"Stale or incompatible frozen result: {path}")
    return data


def collect(args):
    root, run = Path(args.hard_root), Path(args.run_dir)
    _, _, cases = load_hard_protocol(root)
    manifest_hash = sha256(root / "manifest.json")
    static_files = sorted((run / "static").glob("*.json"))
    if not static_files and (run / "static_results.json").exists():
        static_files = [run / "static_results.json"]
    static_aggregate = {}
    for path in static_files:
        data = read(path)
        if data.get("status") != "complete" or data.get("manifest_sha256") != manifest_hash:
            raise ValueError(f"Stale or incompatible static result: {path}")
        static_aggregate.update(data["aggregate"])
    missing_static = set(STATIC_METHODS) - set(static_aggregate)
    if missing_static:
        raise ValueError(f"Missing static methods: {sorted(missing_static)}")
    madqn = read(run / "madqn_results.json")
    if madqn.get("manifest_sha256") != manifest_hash:
        raise ValueError("Comparison output does not match the selected 120 hard cases")
    continuous = validate_frozen_result(root / "direct_mappo" / "seed_23" / "results.json",
                                        manifest_hash, "direct_mappo", 23)
    proposed = validate_frozen_result(root / "full" / "seed_23" / "results.json",
                                      manifest_hash, "full", 23)
    table = []
    for method in STATIC_METHODS:
        table.append(dict(method=method, training_seed=None, unique_cases=len(cases),
                          **static_aggregate[method]))
    table.append(dict(method="continuous_mappo", training_seed=23, unique_cases=len(cases),
                      **continuous["aggregate"]["policy"]))
    table.append(dict(method="discrete_madqn", training_seed=23, unique_cases=len(cases),
                      **madqn["aggregate"]))
    table.append(dict(method="proposed", training_seed=23, unique_cases=len(cases),
                      **proposed["aggregate"]["policy"]))
    payload = {
        "status": "complete", "manifest_sha256": manifest_hash,
        "protocol_note": "Single training seed 23; same 120 fixed evaluation cases.",
        "table": table,
    }
    atomic_json(run / "comparison_summary.json", payload)
    with (run / "comparison_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps(payload, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--hard-root", required=True)
    static = sub.add_parser("static", parents=[common])
    static.add_argument("--output", required=True)
    static.add_argument("--methods", default=",".join(STATIC_METHODS))
    static.add_argument("--workers", type=int, default=12)
    static.add_argument("--mpc-horizon", type=int, default=4)
    static.add_argument("--shooting-candidates", type=int, default=20)
    static.add_argument("--limit", type=int, default=0)
    madqn = sub.add_parser("madqn", parents=[common])
    madqn.add_argument("--output", required=True, type=Path)
    madqn.add_argument("--seed", type=int, default=23)
    madqn.add_argument("--device", default="cuda")
    madqn.add_argument("--episodes", type=int, default=2880)
    madqn.add_argument("--lr", type=float, default=5e-4)
    madqn.add_argument("--gamma", type=float, default=.99)
    madqn.add_argument("--batch-size", type=int, default=256)
    madqn.add_argument("--replay-size", type=int, default=100000)
    madqn.add_argument("--warmup", type=int, default=1024)
    madqn.add_argument("--gradient-steps", type=int, default=2)
    madqn.add_argument("--target-update", type=int, default=50)
    madqn.add_argument("--epsilon-start", type=float, default=1.0)
    madqn.add_argument("--epsilon-end", type=float, default=.05)
    madqn.add_argument("--epsilon-decay", type=int, default=2200)
    madqn.add_argument("--validation-every", type=int, default=240)
    madqn.add_argument("--limit", type=int, default=0)
    madqn.add_argument("--resume", action="store_true")
    collect_parser = sub.add_parser("collect", parents=[common])
    collect_parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    if args.command == "static":
        run_static(args)
    elif args.command == "madqn":
        run_madqn(args)
    else:
        collect(args)


if __name__ == "__main__":
    main()
