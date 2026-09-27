"""Default multi-vehicle intersection scenarios for reproducible evaluation.

No learned-policy outputs enter construction or PG-only selection.
"""
from dataclasses import replace
import hashlib
import json

import numpy as np

from .environment import IntersectionEnv
from .scenarios import Case, Vehicle, ORIGINS, make_case

ANCHOR_SEED = 30_082_001
FAMILIES = ('anchor_perturbation', 'topology_variation', 'independent_arrivals')


def case_from_dict(data):
    return Case(data['seed'], data['density'], data['requested_penetration'],
                tuple(Vehicle(**v) for v in data['vehicles']))


def physical_hash(case):
    return hashlib.sha256(json.dumps(case.to_dict()['vehicles'], sort_keys=True).encode()).hexdigest()


def features(case, cfg):
    env = IntersectionEnv(case, cfg)
    if not all(env.legal_pose(p) for p in env.poses):
        raise ValueError('Illegal initial body position')
    for plan in env.plans:
        if not all(env.legal_pose(p) for p in plan.pose(np.arange(0., plan.length, .7))):
            raise ValueError('Illegal reference route')
    cav_edges = [p for p in env.conflicts if env.cav[p.i] or env.cav[p.j]]
    gaps = [abs(p.station_i / case.vehicles[p.i].speed -
                p.station_j / case.vehicles[p.j].speed) for p in cav_edges]
    degree = [sum(i in (p.i, p.j) for p in env.conflicts) for i in range(env.n)]
    return dict(vehicle_count=env.n, cav_count=int(env.cav.sum()),
                approaches=len({v.origin for v in case.vehicles}),
                conflict_pairs=len(env.conflicts), cav_conflict_pairs=len(cav_edges),
                initial_constant_speed_eta_gap_lt_3=int(sum(g < 3. for g in gaps)),
                max_conflict_degree=max(degree),
                conflict_edges=[[p.i, p.j] for p in env.conflicts])


def candidate(anchor, family, seed):
    rng = np.random.default_rng(seed)
    if family == 'independent_arrivals':
        count = int(rng.integers(6, 9))
        case = make_case(seed, count, float(rng.choice((.5, .75))), 'ordinary')
        arrival = float(rng.uniform(6., 9.))
        vehicles = [replace(v, start_distance=float(np.clip(
            14. + v.speed * (arrival + rng.uniform(-1.5, 1.5)), 28., 75.)))
            for v in case.vehicles]
    else:
        # Retain the three blocking routes, north left turn, and south lead car.
        # Peripheral right turns vary without prescribing any online wait order.
        keep = list(range(8))
        if family == 'topology_variation':
            count = int(rng.integers(6, 9))
            keep = [0, 3, 5, 6, 7] + rng.choice([1, 2, 4], count - 5, replace=False).tolist()
        rotation = int(rng.integers(4))
        rotate = {x: ORIGINS[(i + rotation) % 4] for i, x in enumerate(ORIGINS)}
        common_shift = float(rng.uniform(-5., 5.))
        jitter = 4. if family == 'anchor_perturbation' else 10.
        vehicles = [replace(anchor.vehicles[i],
                    origin=rotate[anchor.vehicles[i].origin],
                    destination=rotate[anchor.vehicles[i].destination],
                    start_distance=float(np.clip(anchor.vehicles[i].start_distance + common_shift
                        + rng.uniform(-jitter, jitter), 28., 75.)),
                    speed=float(np.clip(anchor.vehicles[i].speed + rng.uniform(-.5, .5), 3., 6.)))
                    for i in keep]
        penetration = .75
        if family == 'topology_variation':
            penetration = float(rng.choice((.5, .75)))
            count = max(1, int(np.floor(len(vehicles) * penetration + .5)))
            cav_ids = set(rng.choice(len(vehicles), count, replace=False).tolist())
            vehicles = [replace(v, cav=i in cav_ids) for i, v in enumerate(vehicles)]
        case = Case(seed, 'ordinary', penetration, ())
    used = {}
    for i in sorted(range(len(vehicles)), key=lambda j: vehicles[j].start_distance):
        v = vehicles[i]
        key = v.origin, v.lane
        distance = max(v.start_distance, used.get(key, -1e6) + 13.)
        vehicles[i] = replace(v, start_distance=distance)
        used[key] = distance
    return replace(case, vehicles=tuple(vehicles))


def generate_candidates(anchor, cfg, per_family=120):
    rows, rejected, seen = [], {}, {physical_hash(anchor)}
    for family_index, family in enumerate(FAMILIES):
        accepted = 0
        rejected[family] = 0
        for attempt in range(per_family * 100):
            seed = 70_000_000 + family_index * 1_000_000 + attempt
            case = candidate(anchor, family, seed)
            try:
                feat = features(case, cfg)
            except ValueError:
                rejected[family] += 1
                continue
            fingerprint = physical_hash(case)
            if (feat['approaches'] < 3 or feat['conflict_pairs'] < 5 or
                    feat['cav_conflict_pairs'] < 2 or fingerprint in seen):
                rejected[family] += 1
                continue
            seen.add(fingerprint)
            rows.append(dict(case=case.to_dict(), family=family, features=feat,
                             physical_sha256=fingerprint))
            accepted += 1
            if accepted == per_family:
                break
        if accepted != per_family:
            raise RuntimeError(f'Insufficient valid candidates in {family}: {accepted}')
    return rows, rejected


def select_cases(records, count=120, failure_fraction=.2):
    """Select failures by seed and successes by PG delay, balancing families."""
    if not 0 <= failure_fraction <= 1 or not 0 < count <= len(records):
        raise ValueError('Invalid benchmark size or failure fraction')
    rng = np.random.default_rng(24_092_026)
    failed, success = {}, {}
    for family in FAMILIES:
        group = [r for r in records if r['family'] == family]
        failed[family] = [r for r in group if not r['pg']['success']]
        rng.shuffle(failed[family])
        success[family] = sorted([r for r in group if r['pg']['success']],
                                key=lambda r: (-r['pg']['delay_censored_s'], r['case']['seed']))

    def take(groups, number):
        out = []
        while len(out) < number and any(groups.values()):
            for family in FAMILIES:
                if groups[family] and len(out) < number:
                    out.append(groups[family].pop(0))
        return out

    selected = take(failed, round(count * failure_fraction))
    selected.extend(take(success, count - len(selected)))
    selected.extend(take(failed, count - len(selected)))
    return sorted(selected, key=lambda r: r['case']['seed'])
