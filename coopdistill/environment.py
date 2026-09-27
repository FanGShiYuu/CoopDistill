"""Right-hand-traffic mixed intersection with geometric, causal supervision.

No case-specific release labels, holding nodes or vehicle-ID wait cycles.
Only CAV slots accept policy actions. HDVs follow heterogeneous IDM-like
car following and ETA-based conflict yielding on their own reference routes.
"""

from dataclasses import dataclass
import copy
import math

import numpy as np
from scipy.spatial.distance import cdist

from .geometry import (
    ODReferenceRoutePlan, ResidualRoutePlan, _inbound_point,
    _outbound_point, rectangle_corners, rectangles_overlap, rectangles_overlap_many,
)
from .scenarios import Case, MAX_VEHICLES
from .potential import solve_cav_horizon


@dataclass(frozen=True)
class Config:
    vehicle_length: float = 4.6
    vehicle_width: float = 1.9
    wheelbase: float = 2.8
    stop_gap: float = 1.0
    max_steering_rad: float = math.radians(46.)
    max_lateral_acceleration: float = 2.2
    intersection_half_width: float = 14.
    road_half_width: float = 10.5
    dt: float = 0.1
    decision_steps: int = 5
    horizon_s: float = 60.0
    speed_limit: float = 8.0
    max_acceleration: float = 2.0
    max_brake: float = 4.0
    pg_horizon: int = 8
    pg_dt: float = 0.5
    preview_steps: int = 6
    max_offset: float = 3.0
    transition_m: float = 18.0
    bypass_return_m: float = 18.0
    yield_gap_s: float = 1.5
    near_ttcp_s: float = 1.5
    observation_radius: float = 80.0


@dataclass(frozen=True)
class Conflict:
    i: int
    j: int
    entry_i: float
    exit_i: float
    entry_j: float
    exit_j: float

    @property
    def station_i(self):
        return (self.entry_i + self.exit_i) / 2

    @property
    def station_j(self):
        return (self.entry_j + self.exit_j) / 2


def find_conflicts(plans, specs, cfg):
    samples = [np.arange(0.0, p.length, 0.7) for p in plans]
    points = [p.pose(s) for p, s in zip(plans, samples)]
    result = []
    for i in range(len(plans)):
        for j in range(i+1, len(plans)):
            if specs[i].origin == specs[j].origin and specs[i].lane == specs[j].lane:
                continue  # Same-route following is handled by body-aware obstacle checks.
            near = cdist(points[i][:, :2], points[j][:, :2]) < cfg.vehicle_width + 0.4
            ii, jj = np.nonzero(near)
            if not len(ii):
                continue
            pad = cfg.vehicle_length / 2 + 0.3
            result.append(Conflict(i, j, max(0., samples[i][ii].min()-pad),
                                   samples[i][ii].max()+pad,
                                   max(0., samples[j][jj].min()-pad), samples[j][jj].max()+pad))
    return tuple(result)


class IntersectionEnv:
    obs_dim = 44
    action_dim = 3

    def __init__(self, case: Case, cfg: Config | None = None, record=False):
        self.case, self.cfg = case, cfg or Config()
        if not math.isclose(self.cfg.pg_dt, self.cfg.dt*self.cfg.decision_steps):
            raise ValueError('PG control interval must equal the executed decision interval')
        self.n, self.record = len(case.vehicles), record
        self.cav = np.asarray([v.cav for v in case.vehicles], dtype=bool)
        self.plans = []
        for v in case.vehicles:
            self.plans.append(ODReferenceRoutePlan(
                _inbound_point(v.origin, v.lane, v.start_distance), v.origin, v.lane,
                v.destination, v.lane, v.movement,
                _outbound_point(v.destination, v.lane, 45.0), self.cfg,
            ))
        self.reference_plans = tuple(self.plans)
        self.reference_lengths = np.asarray([p.length for p in self.plans])
        self.conflicts = find_conflicts(self.plans, case.vehicles, self.cfg)
        self.progress = np.zeros(self.n)
        self.speed = np.asarray([v.speed for v in case.vehicles])
        self.acceleration = np.zeros(self.n)
        self.finished = np.zeros(self.n, dtype=bool)
        self.poses = np.asarray([p.pose(0.) for p in self.plans])
        self.distance = np.zeros(self.n)
        self.clearance = np.full(self.n, np.nan)
        self.live_time = np.zeros(self.n)
        self.wait_time = np.zeros(self.n)
        self.next_replan = np.zeros(self.n)
        self.offsets = np.zeros(self.n)
        self.time = 0.0
        self.collision = self.cav_collision = self.boundary = self.dynamic_violation = False
        self.ttcp_min = float("inf")
        self.max_lateral_acceleration = self.max_steering = 0.0
        self.releases = self.accepts = self.proposals = self.solver_failures = 0
        self.history = []
        self._pg_cache = None
        self._limits_cache = None
        if self.collision_pairs():
            raise ValueError("Overlapping initial geometry")
        if record:
            self.history.append({'time': 0., 'poses': self.poses.tolist(),
                                 'speed': self.speed.tolist(), 'finished': self.finished.tolist()})

    def clone(self):
        # Immutable path arrays dominate memory; a shallow copy plus state-array
        # copies avoids duplicating them for every safety preview.
        other = copy.copy(self)
        for key, value in vars(self).items():
            if isinstance(value, np.ndarray):
                setattr(other, key, value.copy())
        other.plans = list(self.plans)
        other.history = []
        other.record = False
        other._limits_cache = None
        other._pg_cache = None
        return other

    @property
    def done(self):
        return bool(self.collision or self.boundary or self.dynamic_violation
                    or self.finished.all() or self.time >= self.cfg.horizon_s-1e-8)

    def mask(self):
        mask = np.zeros(MAX_VEHICLES, dtype=bool)
        mask[:self.n] = self.cav & ~self.finished
        return mask

    def collision_pairs(self):
        return [(i, j) for i in range(self.n) for j in range(i+1, self.n)
                if not self.finished[i] and not self.finished[j]
                and rectangles_overlap(self.poses[i], self.poses[j], self.cfg)]

    def legal_pose(self, pose):
        corners = rectangle_corners(pose, self.cfg)
        central = np.max(np.abs(corners), axis=1) <= self.cfg.intersection_half_width
        horizontal = np.abs(corners[:, 1]) <= self.cfg.road_half_width
        vertical = np.abs(corners[:, 0]) <= self.cfg.road_half_width
        if not np.all(central | horizontal | vertical):
            return False
        x, y, yaw = pose
        gate = self.cfg.intersection_half_width + self.cfg.vehicle_length / 2
        # Outside the junction, prevent crossing into oncoming traffic lanes.
        if abs(x) > gate and (y * math.cos(yaw) >= -self.cfg.vehicle_width/2):
            return False
        if abs(y) > gate and (x * math.sin(yaw) <= self.cfg.vehicle_width/2):
            return False
        return True

    def obstacle_gap(self, idx):
        s = np.arange(0., 30.1, 0.6)
        poses = self.plans[idx].pose(self.progress[idx]+s)
        gap, leader_speed = float("inf"), self.cfg.speed_limit
        for j in range(self.n):
            if j == idx or self.finished[j] or np.linalg.norm(self.poses[j, :2]-self.poses[idx, :2]) > 40:
                continue
            hits = np.flatnonzero(rectangles_overlap_many(poses, self.poses[j], self.cfg, padding=0.12))
            if len(hits) and float(s[hits[0]]) < gap:
                gap = max(0., float(s[hits[0]]) - 0.6)
                alignment = math.cos(self.poses[j, 2]-self.poses[idx, 2])
                leader_speed = max(0., self.speed[j]*alignment)
        return gap, leader_speed

    def safety_limits(self):
        if self._limits_cache is not None:
            return self._limits_cache
        caps = np.full(self.n, self.cfg.speed_limit)
        stops = np.full(self.n, np.inf)
        for i in range(self.n):
            if self.finished[i]:
                caps[i] = 0.
                continue
            look = self.progress[i]+np.linspace(0., max(18., 3*self.speed[i]), 35)
            curvature = np.max(np.abs(self.plans[i].kappa(look)))
            caps[i] = min(caps[i], math.sqrt(0.75*self.cfg.max_lateral_acceleration/max(curvature, 1e-8)))
            gap, _ = self.obstacle_gap(i)
            stops[i] = max(0., gap-self.cfg.stop_gap)
        for pair in self.conflicts:
            i, j = pair.i, pair.j
            if self.finished[i] or self.finished[j]:
                continue
            entries = (pair.entry_i-self.progress[i], pair.entry_j-self.progress[j])
            exits = (pair.exit_i-self.progress[i], pair.exit_j-self.progress[j])
            if min(exits) < 0:
                continue
            eta = (max(entries[0], 0)/max(self.speed[i], 1.), max(entries[1], 0)/max(self.speed[j], 1.))
            occupied = (entries[0] <= 0, entries[1] <= 0)
            # Existing occupancy takes precedence; otherwise order by estimated
            # arrival and physical distance. No ID-based wait-for ring is used.
            priority_i = (not occupied[0], eta[0], entries[0], i) <= (not occupied[1], eta[1], entries[1], j)
            first, second = (0, 1) if priority_i else (1, 0)
            leader, follower = (i, j) if priority_i else (j, i)
            clear_eta = max(exits[first], 0)/max(self.speed[leader], 1.)
            if entries[second] > 0 and eta[second] < clear_eta + self.cfg.yield_gap_s:
                stops[follower] = min(stops[follower], max(0., entries[second]-self.cfg.stop_gap))
        self._limits_cache = caps, stops
        return caps, stops

    def nominal_accelerations(self):
        caps, stops = self.safety_limits()
        result = np.zeros(self.n)
        for i, spec in enumerate(self.case.vehicles):
            if self.finished[i]:
                continue
            gap, leader_v = self.obstacle_gap(i)
            if stops[i] < gap-self.cfg.stop_gap-1e-5:
                gap, leader_v = stops[i]+self.cfg.stop_gap, 0.
            desired = max(0.5, min(spec.desired_speed, caps[i]))
            v = self.speed[i]
            closing = v-leader_v
            desired_gap = 2.0 + max(0., v*spec.headway + v*closing/(2*math.sqrt(1.5*spec.comfortable_brake)))
            result[i] = 1.5*(1-(v/desired)**4 - (desired_gap/max(gap, 0.2))**2)
        return self.project_physical(result)

    def project_physical(self, action):
        caps, stops = self.safety_limits()
        stop_speed = -self.cfg.max_brake*self.cfg.dt + np.sqrt(
            (self.cfg.max_brake*self.cfg.dt)**2 + 2*self.cfg.max_brake*stops)
        upper = (np.minimum(caps, stop_speed)-self.speed)/self.cfg.dt
        result = np.clip(np.minimum(action, upper), -self.cfg.max_brake, self.cfg.max_acceleration)
        result[self.finished] = 0.
        return result

    def pg(self):
        if self._pg_cache is None:
            self._pg_cache = solve_cav_horizon(self)
        return self._pg_cache[0].copy(), dict(self._pg_cache[1])

    def observe(self, include_pg=True):
        pg = self.pg()[0] if include_pg else np.zeros(self.n)
        rows = np.zeros((MAX_VEHICLES, self.obs_dim), dtype=np.float32)
        for i, v in enumerate(self.case.vehicles):
            pose = self.poses[i]
            route = self.reference_plans[i]
            remaining = self.plans[i].length-self.progress[i]
            self_features = [pose[0]/100, pose[1]/100, math.cos(pose[2]), math.sin(pose[2]),
                             self.speed[i]/8, self.acceleration[i]/4, remaining/150,
                             pg[i]/4, self.offsets[i]/3, float(v.cav),
                             route.exit_gate[0]/20, route.exit_gate[1]/20,
                             self.wait_time[i]/60, float(self.finished[i]),
                             float(v.movement == "left"), float(v.movement == "right")]
            rows[i, :16] = self_features
            ordered = sorted((j for j in range(self.n) if j != i and not self.finished[j]),
                             key=lambda j: np.linalg.norm(self.poses[j, :2]-pose[:2]))
            for slot, j in enumerate(ordered[:4]):
                rel = self.poses[j, :2]-pose[:2]
                if np.linalg.norm(rel) > self.cfg.observation_radius:
                    continue
                rows[i, 16+6*slot:22+6*slot] = [rel[0]/80, rel[1]/80,
                    self.speed[j]*math.cos(self.poses[j, 2])/8,
                    self.speed[j]*math.sin(self.poses[j, 2])/8,
                    float(self.cav[j]), 1.]
            rows[i, 40:] = [self.time/self.cfg.horizon_s, self.n/8,
                           float(not self.finished[i]), float(v.cav and not self.finished[i])]
        return rows

    def global_state(self):
        return self.observe().reshape(-1)

    def _make_bypass(self, i, intent):
        if abs(intent) < 0.12 or self.next_replan[i] > self.time or self.finished[i]:
            return None
        plan = self.plans[i]
        remaining = plan.length-self.progress[i]
        if remaining < 2*self.cfg.transition_m+5.:
            return None
        offset = self.cfg.max_offset * intent
        target = plan.pose(self.progress[i]+self.cfg.transition_m)[:2]
        candidate = ResidualRoutePlan(plan, self.progress[i], target, self.cfg, lateral_offset=offset)
        ss = np.arange(0., candidate.length, 0.5)
        if np.max(np.abs(np.arctan(self.cfg.wheelbase*candidate.kappa(ss)))) > self.cfg.max_steering_rad:
            return None
        if not all(self.legal_pose(p) for p in candidate.pose(ss)):
            return None
        return candidate

    def proposal(self, raw, *, lateral=True, absolute=False):
        raw = np.asarray(raw, dtype=float)
        if raw.shape != (MAX_VEHICLES, 3) or not np.isfinite(raw).all():
            raise ValueError("Expected finite padded (8,3) raw policy actions")
        candidate = self.clone()
        pg = self.nominal_accelerations() if absolute else self.pg()[0]
        bounded = np.tanh(raw[:self.n])
        acceleration = pg.copy()
        for i in np.flatnonzero(self.cav & ~self.finished):
            if absolute:
                acceleration[i] = -1.0 + 3.0*bounded[i, 0]
            else:
                # Timing changes desired arrival time at the next conflict;
                # it is not a second copy of acceleration residual.
                distances = [p.station_i-self.progress[i] if p.i == i else p.station_j-self.progress[i]
                             for p in self.conflicts if i in (p.i, p.j)]
                ahead = [d for d in distances if 1. < d < 30.]
                timing_acc = 0.
                if ahead:
                    d = min(ahead)
                    arrival = d/max(self.speed[i], 0.5)
                    target_v = d/max(0.5, arrival+bounded[i, 1])
                    timing_acc = np.clip((target_v-max(self.speed[i], 0.5))/2., -0.8, 0.8)
                acceleration[i] += 1.5*bounded[i, 0] + timing_acc
            if lateral:
                plan = candidate._make_bypass(i, bounded[i, 2])
                if plan is not None:
                    candidate.plans[i] = plan
                    candidate.progress[i] = 0.
                    candidate.offsets[i] = self.cfg.max_offset*bounded[i, 2]
                    candidate.next_replan[i] = self.time+max(5., 2*self.cfg.transition_m/max(self.speed[i], 1.))
                    candidate.releases += 1
        if candidate.releases != self.releases:
            candidate.conflicts = find_conflicts(candidate.plans, self.case.vehicles, self.cfg)
        candidate._pg_cache = candidate._limits_cache = None
        return candidate, acceleration

    def _update_ttcp(self):
        for p in self.conflicts:
            if not (self.cav[p.i] or self.cav[p.j]) or self.finished[p.i] or self.finished[p.j]:
                continue
            di, dj = p.station_i-self.progress[p.i], p.station_j-self.progress[p.j]
            if min(di, dj) < 0 or max(di, dj) > 30 or min(self.speed[p.i], self.speed[p.j]) < 0.2:
                continue
            self.ttcp_min = min(self.ttcp_min, abs(di/self.speed[p.i]-dj/self.speed[p.j]))

    def advance(self, action):
        first_remaining = np.maximum(np.asarray([p.length for p in self.plans])-self.progress, 0.)
        first_finished = self.finished.copy()
        for _ in range(self.cfg.decision_steps):
            if self.done:
                break
            self._limits_cache = None
            nominal = self.nominal_accelerations()
            executable = nominal.copy()
            executable[self.cav] = np.asarray(action)[self.cav]
            executable = self.project_physical(executable)
            old_v = self.speed.copy()
            self.speed = np.clip(old_v + executable*self.cfg.dt, 0, self.cfg.speed_limit)
            self.speed[self.finished] = 0.
            ds = 0.5*(old_v+self.speed)*self.cfg.dt
            ds[self.finished] = 0.
            self.acceleration = (self.speed-old_v)/self.cfg.dt
            # Substeps keep body-overlap checks from skipping a fast crossing.
            for fraction in (0.2, 0.4, 0.6, 0.8, 1.):
                self.poses = np.asarray([p.pose(s+fraction*d) for p, s, d in zip(self.plans, self.progress, ds)])
                hits = self.collision_pairs()
                self.collision |= bool(hits)
                self.cav_collision |= any(self.cav[i] or self.cav[j] for i, j in hits)
                self.boundary |= any(not self.legal_pose(self.poses[i]) for i in range(self.n) if not self.finished[i])
                if hits:
                    break
            self.progress += ds
            self.distance += ds
            live = ~self.finished
            self.live_time += live*self.cfg.dt
            self.wait_time += (live & (self.speed < 0.2))*self.cfg.dt
            self.time += self.cfg.dt
            newly = live & (self.progress >= np.asarray([p.length for p in self.plans])-0.2)
            self.clearance[newly] = self.time
            self.finished |= newly
            kappa = np.asarray([p.kappa(s) for p, s in zip(self.plans, self.progress)])
            lat = float(np.max(np.abs(kappa*self.speed**2)))
            steer = float(np.max(np.abs(np.arctan(kappa*self.cfg.wheelbase))))
            self.max_lateral_acceleration = max(self.max_lateral_acceleration, lat)
            self.max_steering = max(self.max_steering, steer)
            self.dynamic_violation |= lat > self.cfg.max_lateral_acceleration+0.05 or steer > self.cfg.max_steering_rad+1e-6
            self._update_ttcp()
            if self.record:
                self.history.append({"time": self.time, "poses": self.poses.tolist(),
                                     "speed": self.speed.tolist(), "finished": self.finished.tolist()})
            self._limits_cache = self._pg_cache = None
        remaining = np.maximum(np.asarray([p.length for p in self.plans])-self.progress, 0.)
        gain = float(np.sum(first_remaining-remaining))/max(self.n, 1)
        reward = 0.1*gain + 4.*float(np.sum(self.finished & ~first_finished))/self.n
        reward -= 0.05*float(np.mean(~self.finished))
        reward -= 20.*float(self.cav_collision) + 10.*float(self.boundary or self.dynamic_violation)
        return reward

    def preview(self, action=None):
        predicted = self.clone()
        for k in range(self.cfg.preview_steps):
            if predicted.done:
                break
            a = action if k == 0 and action is not None else predicted.pg()[0]
            predicted.advance(a)
        return predicted

    def progress_score(self):
        remaining = np.asarray([p.length for p in self.plans])-self.progress
        # Remaining route distance prevents rewarding a longer detour for its own sake.
        return -float(np.mean(np.maximum(remaining, 0.))) + 10.*float(np.mean(self.finished))

    def step(self, raw=None, *, mode="full", lateral=True):
        if mode == 'direct':
            pg, solver = self.nominal_accelerations(), {'solver_success': True}
        else:
            pg, solver = self.pg()
        self.solver_failures += int(not solver["solver_success"])
        if mode == "pg" or raw is None:
            reward = self.advance(pg)
            return reward, {"accepted": False, "proposed": False}
        candidate, acceleration = self.proposal(raw, lateral=lateral, absolute=mode in ("direct", "absolute"))
        accepted = True
        if mode not in ("no_fallback", "direct"):
            reference = self.preview(pg)
            prediction = candidate.preview(acceleration)
            accepted = (not prediction.cav_collision and not prediction.boundary
                        and not prediction.dynamic_violation
                        and prediction.progress_score() >= reference.progress_score()-1e-6
                        and prediction.collision <= reference.collision)
        self.proposals += 1
        insertion_credit = 0.
        if accepted:
            before = np.mean(np.maximum(np.asarray([p.length for p in self.plans])-self.progress, 0.))
            after = np.mean(np.maximum(np.asarray([p.length for p in candidate.plans])-candidate.progress, 0.))
            insertion_credit = .1*(before-after)
            self.plans, self.conflicts = candidate.plans, candidate.conflicts
            self.progress, self.next_replan = candidate.progress, candidate.next_replan
            self.offsets, self.releases = candidate.offsets, candidate.releases
            self._pg_cache = self._limits_cache = None
            self.accepts += 1
        reward = self.advance(acceleration if accepted else pg)
        reward += insertion_credit
        reward -= 0.03*float(not accepted)
        return reward, {"accepted": accepted, "proposed": True}

    def summary(self):
        average_speed = float(np.sum(self.distance[self.cav])/max(np.sum(self.live_time[self.cav]), 1e-6))
        censored_time = np.where(self.finished, self.clearance, self.cfg.horizon_s)
        delay = np.maximum(censored_time-self.reference_lengths/self.cfg.speed_limit, 0)
        return {
            "seed": self.case.seed, "fingerprint": self.case.fingerprint,
            "vehicle_count": self.n, "cav_count": int(self.cav.sum()),
            "requested_penetration": self.case.requested_penetration,
            "actual_penetration": float(self.cav.mean()), "density": self.case.density,
            "collision": float(self.cav_collision), "total_collision": float(self.collision),
            "success": float(self.finished.all() and not self.collision and not self.boundary and not self.dynamic_violation),
            "cleared_fraction": float(self.finished.mean()), "cav_average_speed": average_speed,
            "delay_censored_s": float(delay.mean()), "unfinished_count": int((~self.finished).sum()),
            "duration_s": float(self.time), "boundary": float(self.boundary),
            "dynamic_violation": float(self.dynamic_violation),
            "episode_ttcp_lt_1p5": float(self.ttcp_min < 1.5),
            "episode_ttcp_lt_0p5": float(self.ttcp_min < 0.5),
            "min_ttcp": self.ttcp_min if np.isfinite(self.ttcp_min) else None,
            "lateral_replans": self.releases, "accept_rate": self.accepts/max(self.proposals, 1),
            "pg_solver_failures": self.solver_failures,
            "max_lateral_acceleration": self.max_lateral_acceleration,
            "max_steering_deg": math.degrees(self.max_steering),
        }
