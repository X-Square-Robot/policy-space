import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "policies/wallx/wallx_public_api"
spec = importlib.util.spec_from_file_location("wallx_api_tested", ROOT / "model.py")
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


class Tests(unittest.TestCase):
    def test_gripper_roundtrip_and_shapes(self):
        for scale in (1.526, 5.5):
            rows = np.ones((32,14), np.float32) * .25
            named = api.named_rows(rows, scale)
            np.testing.assert_allclose(api.action_rows({'actions':named}, scale), rows)
            named['left_ee']['gripper'][0] = scale + 1
            with self.assertRaises(ValueError):
                api.action_rows({'actions':named}, scale)

    def test_padding_range_and_valid_row_checks(self):
        for scale in (1.526, 5.5):
            rows = np.ones((32,14), np.float32) * .25
            named = api.named_rows(rows, scale)
            named['right_ee']['gripper'][31] = -0.0032
            result = api.action_rows({'actions':named}, scale)
            np.testing.assert_allclose(result[:31], rows[:31])
            self.assertLess(result[31,13], 0)  # Padding preserved, never executed.
            named['right_ee']['gripper'][30] = -0.0032
            with self.assertRaises(ValueError):
                api.action_rows({'actions':named}, scale)
            named['right_ee']['gripper'][30] = scale + .1
            with self.assertRaises(ValueError):
                api.action_rows({'actions':named}, scale)
            named['right_ee']['gripper'][30] = 0
            named['right_ee']['gripper'][31] = float('nan')
            with self.assertRaises(ValueError):
                api.action_rows({'actions':named}, scale)

    def test_episode_clears_cached_observation(self):
        model=api.Model({'gripper':'H'})
        model.ingest_observation({'dummy':True})
        old=model.session
        model.initialize_episode()
        self.assertIsNone(model.latest)
        self.assertNotEqual(old,model.session)
        self.assertEqual(model.calls,0)
