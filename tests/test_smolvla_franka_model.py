"""Franka wire/model gripper boundary regressions without loading weights."""
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from policies.smolvla.smolvla_franka_joint.model import CAMERA_MAP, Model, _interpolate_joint_actions, _interpolated_horizon


def model_shell():
    model = Model.__new__(Model)
    model._camera_map = dict(CAMERA_MAP)
    model._instruction = "pick up the object"
    model._require_instruction = True
    model._action_dim = 8
    model._action_horizon = 50
    model._native_action_horizon = 50
    model._interpolation_multiplier = 1.0
    model._latest = None
    model._device = torch.device("cpu")
    model._checkpoint = Path("checkpoint")
    return model


def observation(gripper, layout="split"):
    state = np.array([0, -0.569, 0, -2.81, 0, 3.037, 0.741, gripper], dtype=np.float32)
    if layout == "split":
        fields = {"joint_position": state[:7], "gripper_position": state[7:]}
    else:
        fields = {layout: state}
    return {"images": {k: np.zeros((4, 5, 3), dtype=np.uint8) for k in CAMERA_MAP},
            "state": fields, "instruction": "pick up the object"}, state


class SmolVLAFrankaGripperTest(unittest.TestCase):
    def test_sim_state_converts_before_preprocessing_without_mutating_request(self):
        for layout in ("split", "observation.state", "state"):
            for wire, model_expected in ((0.0, 1.0), (1.0, 0.0), (0.25, 0.75)):
                with self.subTest(layout=layout, wire=wire):
                    model = model_shell()
                    obs, source = observation(wire, layout)
                    before = source.copy()
                    model.ingest_observation(obs)
                    actual = model._latest["observation"]["observation.state"]
                    self.assertEqual(float(actual[7]), model_expected)
                    np.testing.assert_array_equal(actual[:7], before[:7])
                    np.testing.assert_array_equal(source, before)
                    self.assertFalse(np.shares_memory(actual, source))

    def test_full_pipeline_converts_after_unnormalizing(self):
        model = model_shell()
        obs, _ = observation(0.0)
        model.ingest_observation(obs)
        captured = []
        def prepare(frame, device, task):
            captured.append(frame["observation.state"].copy())
            return frame
        model._prepare_observation = prepare
        model._preprocess = lambda frame: frame
        normalized = torch.zeros((1, 50, 8), dtype=torch.float32)
        physical = torch.zeros_like(normalized)
        physical[0, :, :7] = torch.arange(7) / 10
        # Recorded-model-like outputs, including flow noise beyond [0, 1].
        physical[0, :5, 7] = torch.tensor([1.0, 0.0, 0.25, 1.01, -0.01])
        before = physical.clone()
        model._policy = SimpleNamespace(predict_action_chunk=lambda batch: normalized)
        model._postprocess = lambda actions: physical
        wire = model.infer_actions()
        self.assertEqual(float(captured[0][7]), 1.0)
        self.assertEqual(wire.shape, (50, 8))
        self.assertEqual(wire.dtype, np.float32)
        self.assertTrue(wire.flags.c_contiguous)
        np.testing.assert_array_equal(wire[:5, 7], [0.0, 1.0, 0.75, 0.0, 1.0])
        np.testing.assert_array_equal(wire[:, :7], before[0, :, :7].numpy())
        np.testing.assert_array_equal(physical.numpy(), before.numpy())
        # Franka's executed finger targets: model open must command 0.04 m.
        targets = np.where(wire[:2, 7] > 0.5, 0.0, 0.04)
        np.testing.assert_allclose(targets, [0.04, 0.0])

    def test_invalid_state_still_fails_before_conversion(self):
        model = model_shell()
        obs, _ = observation(float("nan"))
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            model.ingest_observation(obs)

    def test_runtime_identifies_the_gripper_adapter(self):
        metadata = model_shell().runtime_metadata()
        self.assertEqual(metadata.get("smolvla_gripper_adapter"), "franka_closed01_to_model_open01_v1")
        self.assertEqual(metadata.get("smolvla_model_gripper_convention"), "0_closed_1_open")
        self.assertEqual(metadata.get("smolvla_wire_gripper_convention"), "0_open_1_closed")


    def test_interpolation_preserves_joint_units_and_gripper_switch_time(self):
        native = np.zeros((50, 8), dtype=np.float32)
        # Deliberately non-unit joint values detect accidental quaternion normalization.
        native[:, :7] = np.arange(50)[:, None] * np.arange(1, 8)[None, :] / 10
        native[20:, 7] = 1.0
        before = native.copy()
        out = _interpolate_joint_actions(native, 1.5)
        self.assertEqual(out.shape, (75, 8))
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(out.flags.c_contiguous)
        np.testing.assert_array_equal(native, before)
        np.testing.assert_array_equal(out[0], native[0])
        np.testing.assert_array_equal(out[-1], native[-1])
        # Native tick2 lands exactly on output tick3; tick20 -> output tick30.
        np.testing.assert_allclose(out[3, :7], native[2, :7])
        np.testing.assert_allclose(out[2, :7], (native[1, :7] * 2 + native[2, :7]) / 3, rtol=1e-6)
        np.testing.assert_array_equal(out[:30, 7], 0.0)
        np.testing.assert_array_equal(out[30:, 7], 1.0)

    def test_interpolated_pipeline_validates_native_shape_and_reports_wire_shape(self):
        model = model_shell()
        model._interpolation_multiplier = 1.5
        model._action_horizon = 75
        obs, _ = observation(0.0)
        model.ingest_observation(obs)
        model._prepare_observation = lambda frame, device, task: frame
        model._preprocess = lambda frame: frame
        normalized = torch.zeros((1, 50, 8), dtype=torch.float32)
        physical = torch.zeros_like(normalized)
        physical[0, :, 3:7] = torch.tensor([-2.8, 0.2, 3.0, 0.74])
        physical[0, :20, 7] = 1.0  # model opening -> wire0 for first30 rows
        model._policy = SimpleNamespace(predict_action_chunk=lambda batch: normalized)
        model._postprocess = lambda actions: physical
        out = model.infer_actions()
        self.assertEqual(out.shape, (75, 8))
        np.testing.assert_allclose(out[:, 3:7], physical[0, 0, 3:7].numpy()[None, :].repeat(75, axis=0))
        np.testing.assert_array_equal(out[:30, 7], 0.0)
        np.testing.assert_array_equal(out[30:, 7], 1.0)
        self.assertEqual(model.runtime_metadata()["smolvla_action_shape"], [75, 8])
        self.assertEqual(model.runtime_metadata()["smolvla_native_action_shape"], [50, 8])
        model._postprocess = lambda actions: physical[:, :49]
        with self.assertRaisesRegex(RuntimeError, "output must have shape"):
            model.infer_actions()

    def test_invalid_interpolation_timing_is_rejected(self):
        for factor in [0.0, 0.5, float("nan"), float("inf"), 1.25]:
            with self.subTest(factor=factor), self.assertRaises(ValueError):
                _interpolated_horizon(50, factor)


if __name__ == "__main__":
    unittest.main()
