"""DGame-structured horizon objective, optimized over CAV accelerations only.

Ports the active r1/r4 and discount structure of Vehicle.convex_potential_gurobi_expr.
Geometric pairwise conflict stations replace its dis2cp matrix. Original r2 and
r3 were inactive; this port also leaves them inactive. The feasible set is new:
speed, acceleration, obstacle and conflict-reservation bounds. HiGHS solves the
linear horizon program; this is not a claim of reproducing the original solver
or proving an exact potential-game equilibrium under arbitrary HDV responses.
"""

import numpy as np
import warnings
from scipy.optimize import linprog, OptimizeWarning


def dgame_coefficients(env, horizon: int, dt: float) -> np.ndarray:
    n = env.n
    discount = 0.9 ** np.arange(horizon)
    travel = np.tril(np.ones((horizon, horizon)))
    travel = travel @ travel * dt**2
    coefficients = np.tile(discount, (n, 1))
    for pair in env.conflicts:
        i, j = pair.i, pair.j
        if env.finished[i] or env.finished[j]:
            continue
        di = pair.station_i - env.progress[i]
        dj = pair.station_j - env.progress[j]
        if not (0 < di < 30.0 and 0 < dj < 30.0):
            continue
        # Keep both directed terms explicitly, including equal-arrival ties.
        for a, b, da, db in ((i, j, di, dj), (j, i, dj, di)):
            sign = 1.0 if da / max(env.speed[a], 0.1) > db / max(env.speed[b], 0.1) else -1.0
            multiple = (30.0 - abs(da - db)) * (30.0 - db)
            influence = multiple * sign * (discount @ travel)
            coefficients[a] -= influence
            coefficients[b] += influence
    return coefficients


def solve_cav_horizon(env) -> tuple[np.ndarray, dict]:
    cfg, n = env.cfg, env.n
    h, dt = cfg.pg_horizon, cfg.pg_dt
    nominal = env.nominal_accelerations()
    ids = np.flatnonzero(env.cav & ~env.finished)
    if not ids.size:
        return nominal, {"solver_success": True, "optimized_cavs": 0}
    coefficients = dgame_coefficients(env, h, dt)
    lower = np.tril(np.ones((h, h))) * dt
    # Feasibility uses trapezoidal kinematics, matching the physical executor;
    # the original DGame utility's semi-implicit predictor is preserved above.
    travel = (np.tril(np.ones((h, h))) @ np.tril(np.ones((h, h)))
              - 0.5*np.tril(np.ones((h, h)))) * dt**2
    variable_count = len(ids) * h
    objective = -coefficients[ids].reshape(-1)
    # A global positive scale preserves the maximizer and conditions the LP.
    objective /= max(1.0, float(np.max(np.abs(objective))))
    constraints, limits = [], []
    caps, stops = env.safety_limits()
    bounds = []
    for k, i in enumerate(ids):
        block = slice(k*h, (k+1)*h)
        for t in range(h):
            row = np.zeros(variable_count)
            row[block] = lower[t]
            constraints.extend((row, -row))
            limits.extend((max(caps[i], env.speed[i]) - env.speed[i], env.speed[i]))
            if np.isfinite(stops[i]):
                row = np.zeros(variable_count)
                row[block] = travel[t]
                constraints.append(row)
                limits.append(max(0.0, stops[i]) - env.speed[i]*dt*(t+1))
            bounds.append((-cfg.max_brake, cfg.max_acceleration))
    with warnings.catch_warnings():
        # SciPy forwards this solver-specific option to HiGHS; keep each
        # independent rollout worker single-threaded rather than oversubscribed.
        warnings.filterwarnings('ignore', message='Unrecognized options detected.*', category=OptimizeWarning)
        solution = linprog(objective, A_ub=np.asarray(constraints), b_ub=np.asarray(limits),
                           bounds=bounds, method="highs", options={'threads': 1})
    action = nominal.copy()
    if solution.success:
        action[ids] = solution.x.reshape(-1, h)[:, 0]
    # Both PG and learned proposals share a physical braking/curvature envelope.
    action = env.project_physical(action)
    return action, {
        "solver_success": bool(solution.success), "status": int(solution.status),
        "optimized_cavs": len(ids), "horizon": h,
        "objective": float(solution.fun) if solution.success else None,
    }
