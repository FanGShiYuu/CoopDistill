"""Contract checks for the new benchmark, without training or remote services."""
import unittest
from dataclasses import replace
import numpy as np

from .environment import Config, IntersectionEnv
from .scenarios import make_case, suite, MAX_VEHICLES
from .potential import dgame_coefficients


class Contracts(unittest.TestCase):
    def test_random_geometry_and_split(self):
        cases = [make_case(s) for s in range(100)]
        self.assertGreater(len({c.fingerprint for c in cases}), 19)
        self.assertEqual({len(c.vehicles) for c in cases}, {4, 5, 6, 7, 8})
        self.assertEqual({v.origin for c in cases for v in c.vehicles}, set('WSEN'))
        for case in cases:
            env = IntersectionEnv(case)
            self.assertFalse(env.collision_pairs())
            for plan in env.plans:
                for pose in plan.pose(np.linspace(0., plan.length, 50)):
                    self.assertTrue(env.legal_pose(pose), (case.seed, pose))
        self.assertFalse({c.seed for c in suite('validation')} & {c.seed for c in suite('test')})

    def test_zero_action_and_hdv_isolation(self):
        env = IntersectionEnv(make_case(71))
        pg, status = env.pg()
        self.assertEqual(status['optimized_cavs'], int(env.cav.sum()))
        zero = np.zeros((MAX_VEHICLES, 3))
        proposal, acceleration = env.proposal(zero)
        np.testing.assert_allclose(pg, acceleration)
        self.assertEqual(proposal.releases, 0)
        raw = zero.copy()
        raw[:env.n][~env.cav] = 10.
        ignored, acceleration = env.proposal(raw)
        np.testing.assert_allclose(acceleration, pg)
        self.assertEqual(ignored.releases, 0)
        a, b = env.clone(), env.clone()
        a.step(mode='pg')
        b.step(zero)
        np.testing.assert_allclose(a.poses, b.poses, atol=1e-10)

    def test_preview_is_nonmutating(self):
        env = IntersectionEnv(make_case(93), Config(preview_steps=2))
        before = env.poses.copy()
        state = env.observe().copy()
        env.preview()
        np.testing.assert_array_equal(env.poses, before)
        np.testing.assert_array_equal(env.observe(), state)
        self.assertEqual(env.time, 0.)
        self.assertEqual(env.releases, 0)

    def test_objective_matches_directed_original_formula(self):
        env = IntersectionEnv(make_case(19, count=8, density='dense'))
        h, dt = 4, .3
        rng = np.random.default_rng(12)
        actions = rng.normal(size=(env.n, h))
        def objective(a):
            value = 0.
            for t in range(h):
                r1 = a[:, t].sum()
                r4 = 0.
                for pair in env.conflicts:
                    for i, j, si, sj in ((pair.i, pair.j, pair.station_i, pair.station_j),
                                         (pair.j, pair.i, pair.station_j, pair.station_i)):
                        di, dj = si-env.progress[i], sj-env.progress[j]
                        if not (0 < di < 30 and 0 < dj < 30):
                            continue
                        sign = 1. if di/max(env.speed[i], .1) > dj/max(env.speed[j], .1) else -1.
                        coeff = (30-abs(di-dj))*(30-dj)
                        steps = np.arange(t+1, 0, -1)
                        pi = di-env.speed[i]*dt*(t+1)-dt**2*np.dot(a[i, :t+1], steps)
                        pj = dj-env.speed[j]*dt*(t+1)-dt**2*np.dot(a[j, :t+1], steps)
                        r4 += (pi-pj)*coeff*sign
                value += .9**t*(r1+r4)
            return value
        lhs = objective(actions)-objective(np.zeros_like(actions))
        rhs = np.sum(dgame_coefficients(env, h, dt)*actions)
        self.assertAlmostEqual(lhs, rhs, places=7)

    def test_finished_agents_are_masked(self):
        env = IntersectionEnv(make_case(71))
        i = np.flatnonzero(env.cav)[0]
        env.finished[i] = True
        self.assertFalse(env.mask()[i])
        self.assertFalse(env.mask()[env.n:].any())

    def test_bypass_continuity(self):
        found = 0
        for s in range(5):
            env = IntersectionEnv(make_case(s))
            for i in np.flatnonzero(env.cav):
                plan = env._make_bypass(i, .6)
                if plan is None:
                    continue
                found += 1
                np.testing.assert_allclose(plan.pose(0.)[:2], env.poses[i, :2], atol=1e-5)
                yaw_delta = np.arctan2(np.sin(plan.pose(0.)[2]-env.poses[i, 2]),
                                      np.cos(plan.pose(0.)[2]-env.poses[i, 2]))
                self.assertLess(abs(yaw_delta), .01)
                self.assertLess(np.max(np.abs(np.arctan(env.cfg.wheelbase*plan.curvature))), env.cfg.max_steering_rad)
        self.assertGreater(found, 0)

    def test_masked_mappo_updates_and_gae(self):
        import torch
        from .trainer import Actor, Critic, gae, rollout, update
        advantage, returns = gae(np.array([1., 2.]), np.array([.5, .4]), gamma=1., lam=1.)
        np.testing.assert_allclose(returns, [3., 2.])
        torch.set_num_threads(1)
        actor, critic = Actor(), Critic()
        before = [p.detach().clone() for p in actor.parameters()]
        case = make_case(84)
        batch = rollout((case, Config(horizon_s=1., preview_steps=1), 'no_fallback',
                         actor.state_dict(), 981, False, False))
        stats = update(actor, critic, torch.optim.Adam(actor.parameters(), lr=.001),
                       torch.optim.Adam(critic.parameters(), lr=.001), [batch], 'no_fallback',
                       torch.device('cpu'), 2, 32)
        self.assertTrue(all(np.isfinite(v) for v in stats.values()))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, actor.parameters())))

    def test_bad_candidate_uses_real_time_pg_not_future_oracle(self):
        from unittest.mock import patch
        env = IntersectionEnv(make_case(10), Config(preview_steps=1))
        reference = env.clone()
        bad = env.clone()
        bad.boundary = True
        with patch.object(IntersectionEnv, 'preview', side_effect=[reference, bad]):
            _, info = env.step(np.ones((8, 3)))
        self.assertFalse(info['accepted'])
        reference.step(mode='pg')
        np.testing.assert_allclose(env.poses, reference.poses)

    def test_independent_mappo_does_not_call_pg(self):
        from unittest.mock import patch
        env = IntersectionEnv(make_case(12))
        with patch.object(IntersectionEnv, 'pg', side_effect=AssertionError('Direct MAPPO called PG')):
            self.assertTrue(np.all(env.observe(include_pg=False)[:, 7] == 0.))
            env.step(np.zeros((8, 3)), mode='direct')

    def test_pg_interval_must_match_execution(self):
        with self.assertRaises(ValueError):
            IntersectionEnv(make_case(12), Config(pg_dt=.3))


if __name__ == '__main__':
    unittest.main()
