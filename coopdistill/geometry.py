"""Frozen project route/rectangle primitives; no legacy environment or controller.
Source SHA256: 499a5492bd9a0c9d79da052e1cd4d9c781de8c39865e7d0d0bccd08b2bedefcc
"""
from __future__ import annotations
import math
import numpy as np
LANE_CENTERS_M = (1.75, 5.25, 8.75)

INBOUND_HEADINGS = {'W': 0.0, 'S': math.pi / 2, 'E': math.pi, 'N': -math.pi / 2}

OUTBOUND_HEADINGS = {'W': math.pi, 'S': -math.pi / 2, 'E': 0.0, 'N': math.pi / 2}

def _inbound_point(approach: str, lane: int, longitudinal: float) -> np.ndarray:
    offset = LANE_CENTERS_M[lane]
    return {'W': np.array([-longitudinal, -offset]), 'E': np.array([longitudinal, offset]), 'S': np.array([offset, -longitudinal]), 'N': np.array([-offset, longitudinal])}[approach]

def _outbound_point(destination: str, lane: int, longitudinal: float) -> np.ndarray:
    offset = LANE_CENTERS_M[lane]
    return {'W': np.array([-longitudinal, offset]), 'E': np.array([longitudinal, -offset]), 'S': np.array([-offset, -longitudinal]), 'N': np.array([offset, longitudinal])}[destination]

def smoothstep5(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5

class RoutePlan:
    """Arc-length path through an occupied conflict zone or around it."""

    def __init__(self, start: np.ndarray, target: np.ndarray, exit_point: np.ndarray, exit_heading: float, cfg, *, lateral_offset: float=0.0):
        start = np.asarray(start, dtype=float)
        target = np.asarray(target, dtype=float)
        exit_point = np.asarray(exit_point, dtype=float)
        vector = target - start
        target_distance = float(np.linalg.norm(vector))
        if target_distance < 2.0:
            raise ValueError('Route target is too close to the start')
        direction = vector / target_distance
        normal = np.array([-direction[1], direction[0]])
        return_distance = cfg.bypass_return_m
        q = np.linspace(0.0, target_distance + return_distance, 501)
        rise = smoothstep5(q / target_distance)
        fall = smoothstep5((q - target_distance) / return_distance)
        lateral = lateral_offset * (rise - fall)
        first = start + q[:, None] * direction + lateral[:, None] * normal
        join = first[-1]
        exit_direction = np.array([math.cos(exit_heading), math.sin(exit_heading)])
        curve_distance = float(np.linalg.norm(exit_point - join))
        control = min(12.0, 0.35 * curve_distance)
        p0 = join
        p1 = join + control * direction
        p3 = exit_point
        p2 = exit_point - control * exit_direction
        u = np.linspace(0.0, 1.0, 701)[1:]
        second = ((1 - u) ** 3)[:, None] * p0 + (3 * (1 - u) ** 2 * u)[:, None] * p1
        second += (3 * (1 - u) * u ** 2)[:, None] * p2 + (u ** 3)[:, None] * p3
        points = np.vstack((first, second))
        self._set_points(points, target_distance=target_distance, lateral_offset=lateral_offset)

    def _set_points(self, points: np.ndarray, *, target_distance: float, lateral_offset: float) -> None:
        arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
        dx = np.gradient(points[:, 0], arc, edge_order=2)
        dy = np.gradient(points[:, 1], arc, edge_order=2)
        yaw = np.unwrap(np.arctan2(dy, dx))
        curvature = np.gradient(yaw, arc, edge_order=2)
        self.points = points
        self.arc = arc
        self.yaw = yaw
        self.curvature = curvature
        self.target_distance = target_distance
        self.lateral_offset = float(lateral_offset)

    @property
    def length(self) -> float:
        return float(self.arc[-1])

    def pose(self, distance) -> np.ndarray:
        distance = np.asarray(distance)
        x = np.interp(distance, self.arc, self.points[:, 0])
        y = np.interp(distance, self.arc, self.points[:, 1])
        yaw = np.interp(distance, self.arc, self.yaw)
        return np.stack((x, y, yaw), axis=-1)

    def kappa(self, distance):
        return np.interp(distance, self.arc, self.curvature)

class ODReferenceRoutePlan(RoutePlan):
    """Lane-consistent reference path from a legal entry to a legal exit."""

    def __init__(self, start: np.ndarray, origin: str, origin_lane: int, destination: str, destination_lane: int, movement: str, exit_point: np.ndarray, cfg):
        start = np.asarray(start, dtype=float)
        exit_point = np.asarray(exit_point, dtype=float)
        gate = cfg.intersection_half_width
        entry = _inbound_point(origin, origin_lane, gate)
        departure = _outbound_point(destination, destination_lane, gate)
        inbound_heading = INBOUND_HEADINGS[origin]
        outbound_heading = OUTBOUND_HEADINGS[destination]
        inbound_direction = np.array([math.cos(inbound_heading), math.sin(inbound_heading)])
        outbound_direction = np.array([math.cos(outbound_heading), math.sin(outbound_heading)])
        approach = np.linspace(start, entry, 401)
        if movement == 'straight':
            junction = np.linspace(entry, departure, 501)
        else:
            turn_radius = float(np.max(np.abs(departure - entry)))
            control = 0.55228475 * turn_radius
            p0 = entry
            p1 = entry + control * inbound_direction
            p2 = departure - control * outbound_direction
            p3 = departure
            u = np.linspace(0.0, 1.0, 601)
            junction = ((1 - u) ** 3)[:, None] * p0 + (3 * (1 - u) ** 2 * u)[:, None] * p1
            junction += (3 * (1 - u) * u ** 2)[:, None] * p2 + (u ** 3)[:, None] * p3
        outbound = np.linspace(departure, exit_point, 401)
        points = np.vstack((approach[:-1], junction[:-1], outbound))
        self._set_points(points, target_distance=0.0, lateral_offset=0.0)
        self.entry_point = entry
        self.exit_gate = departure
        self.origin = origin
        self.destination = destination
        self.origin_lane = origin_lane
        self.destination_lane = destination_lane
        self.movement = movement

class ResidualRoutePlan(RoutePlan):
    """Frenet residual over the remaining PG route with C1 continuity."""

    def __init__(self, base_plan: RoutePlan, start_progress: float, target: np.ndarray, cfg, *, lateral_offset: float):
        distances = np.linspace(float(start_progress), base_plan.length, 1201)
        base_poses = base_plan.pose(distances)
        points = base_poses[:, :2].copy()
        remaining = distances - float(start_progress)
        target_idx = int(np.argmin(np.linalg.norm(points - np.asarray(target), axis=1)))
        target_distance = max(float(remaining[target_idx]), 2.0)
        rise = smoothstep5(remaining / target_distance)
        fall = smoothstep5((remaining - target_distance) / cfg.bypass_return_m)
        lateral = lateral_offset * (rise - fall)
        normals = np.column_stack((-np.sin(base_poses[:, 2]), np.cos(base_poses[:, 2])))
        points += lateral[:, None] * normals
        self._set_points(points, target_distance=target_distance, lateral_offset=lateral_offset)

def rectangle_corners(pose: np.ndarray, cfg, padding: float=0.0) -> np.ndarray:
    half_length = cfg.vehicle_length / 2 + padding
    half_width = cfg.vehicle_width / 2 + padding
    local = np.array([[half_length, half_width], [half_length, -half_width], [-half_length, -half_width], [-half_length, half_width]])
    c, s = (math.cos(float(pose[2])), math.sin(float(pose[2])))
    rotation = np.array([[c, -s], [s, c]])
    return local @ rotation.T + pose[:2]

def rectangles_overlap(a: np.ndarray, b: np.ndarray, cfg, padding: float=0.0) -> bool:
    af = np.array([math.cos(a[2]), math.sin(a[2])])
    aside = np.array([-af[1], af[0]])
    bf = np.array([math.cos(b[2]), math.sin(b[2])])
    bside = np.array([-bf[1], bf[0]])
    delta = b[:2] - a[:2]
    hl = cfg.vehicle_length / 2 + padding
    hw = cfg.vehicle_width / 2 + padding
    for axis in (af, aside, bf, bside):
        ra = hl * abs(float(af @ axis)) + hw * abs(float(aside @ axis))
        rb = hl * abs(float(bf @ axis)) + hw * abs(float(bside @ axis))
        if abs(float(delta @ axis)) > ra + rb:
            return False
    return True

def rectangles_overlap_many(poses: np.ndarray, obstacle: np.ndarray, cfg, padding: float=0.0) -> np.ndarray:
    """Vectorized SAT test between path poses and one occupied rectangle."""
    poses = np.atleast_2d(poses)
    af = np.column_stack((np.cos(poses[:, 2]), np.sin(poses[:, 2])))
    aside = np.column_stack((-af[:, 1], af[:, 0]))
    bf = np.array([math.cos(obstacle[2]), math.sin(obstacle[2])])
    bside = np.array([-bf[1], bf[0]])
    delta = obstacle[:2] - poses[:, :2]
    hl = cfg.vehicle_length / 2 + padding
    hw = cfg.vehicle_width / 2 + padding
    overlap = np.ones(len(poses), dtype=bool)
    for axis, varying in ((af, True), (aside, True), (np.broadcast_to(bf, af.shape), False), (np.broadcast_to(bside, af.shape), False)):
        projection = np.abs(np.sum(delta * axis, axis=1))
        ra = hl * np.abs(np.sum(af * axis, axis=1)) + hw * np.abs(np.sum(aside * axis, axis=1))
        rb = hl * np.abs(axis @ bf) + hw * np.abs(axis @ bside)
        overlap &= projection <= ra + rb
    return overlap
