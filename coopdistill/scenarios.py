"""Serializable scenarios with independent random OD, positions and CAV masks."""

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

ORIGINS = ("W", "S", "E", "N")
MOVEMENTS = ("left", "straight", "right")
DESTINATIONS = {
    "W": ("N", "E", "S"), "S": ("W", "N", "E"),
    "E": ("S", "W", "N"), "N": ("E", "S", "W"),
}
MAX_VEHICLES = 8


@dataclass(frozen=True)
class Vehicle:
    origin: str
    destination: str
    movement: str
    lane: int
    start_distance: float
    speed: float
    cav: bool
    desired_speed: float
    headway: float
    comfortable_brake: float


@dataclass(frozen=True)
class Case:
    seed: int
    density: str
    requested_penetration: float
    vehicles: tuple[Vehicle, ...]

    def to_dict(self):
        return asdict(self)

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def make_case(seed: int, count: int | None = None, penetration: float | None = None,
              density: str | None = None) -> Case:
    rng = np.random.default_rng(seed)
    count = int(rng.integers(4, 9)) if count is None else count
    penetration = float(rng.choice([0.25, 0.5, 0.75])) if penetration is None else penetration
    density = str(rng.choice(["ordinary", "dense"])) if density is None else density
    if not 4 <= count <= MAX_VEHICLES or not 0 < penetration <= 1:
        raise ValueError("Expect 4--8 vehicles and penetration in (0, 1]")
    if density not in ("ordinary", "dense"):
        raise ValueError(density)
    cav_count = max(1, min(count, int(np.floor(count * penetration + 0.5))))
    cav_ids = set(rng.choice(count, cav_count, replace=False).tolist())
    positions: dict[tuple[str, int], list[float]] = {}
    vehicles = []
    for idx in range(count):
        origin = str(rng.choice(ORIGINS))
        move = int(rng.choice(3, p=[0.30, 0.45, 0.25]))
        movement, destination = MOVEMENTS[move], DESTINATIONS[origin][move]
        low, high = (23.0, 40.0) if density == "dense" else (28.0, 75.0)
        distance = float(rng.uniform(low, high))
        used = positions.setdefault((origin, move), [])
        # All cars are present at t=0. Reject overlapping/unsafe same-lane starts.
        while any(abs(distance - previous) < 13.0 for previous in used):
            distance += 13.0
        used.append(distance)
        speed = float(rng.uniform(2.5, 4.0) if density == "dense" else rng.uniform(3.0, 6.0))
        vehicles.append(Vehicle(
            origin, destination, movement, move, distance, speed, idx in cav_ids,
            float(rng.uniform(5.5, 7.5)), float(rng.uniform(1.1, 1.9)),
            float(rng.uniform(1.5, 2.5)),
        ))
    return Case(seed, density, penetration, tuple(vehicles))


def suite(split: str, per_stratum: int = 2) -> list[Case]:
    base = {"validation": 20_000_000, "test": 30_000_000}[split]
    cases = []
    for count in range(4, 9):
        for pen_idx, penetration in enumerate((0.25, 0.5, 0.75)):
            for dense_idx, density in enumerate(("ordinary", "dense")):
                for rep in range(per_stratum):
                    seed = base + count * 10000 + pen_idx * 1000 + dense_idx * 100 + rep
                    cases.append(make_case(seed, count, penetration, density))
    return cases
