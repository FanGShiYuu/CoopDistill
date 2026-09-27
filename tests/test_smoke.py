from pathlib import Path
import unittest

import torch

from coopdistill.comparison import load_hard_protocol
from coopdistill.environment import Config
from coopdistill.scenarios import make_case
from coopdistill.trainer import Actor, rollout


ROOT = Path(__file__).resolve().parents[1]


class PublicReleaseSmokeTest(unittest.TestCase):
    def test_default_benchmark_integrity(self):
        _, manifest, cases = load_hard_protocol(ROOT / "benchmarks/default")
        self.assertEqual(len(cases), 120)
        self.assertEqual(manifest["selected_count"], 120)
        self.assertTrue(all(6 <= len(case.vehicles) <= 8 for case in cases))

    def test_checkpoint_matches_actor(self):
        actor = Actor()
        path = ROOT / "checkpoints/coopdistill_seed23.pt"
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
        actor.load_state_dict(state)

    def test_short_pg_rollout(self):
        case = make_case(12345)
        result = rollout((case, Config(horizon_s=1.0, preview_steps=1), "pg", None,
                          case.seed, True, False))
        self.assertFalse(result["summary"]["collision"])
        self.assertEqual(result["summary"]["vehicle_count"], len(case.vehicles))


if __name__ == "__main__":
    unittest.main()
