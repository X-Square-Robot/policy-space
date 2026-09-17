"""FastWAM bundle wire-contract tests without loading the CUDA model."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from policies.fastwam.fastwam_artixon_arm_6a_joint.model import (
    ACTION_HORIZON,
    CAMERA_KEYS,
    Model,
    pack_x2real_action,
)
from policy_space.server import load_policy_definition


def _model_shell() -> Model:
    model = Model.__new__(Model)
    model._latest_obs = None
    model._episode_id = "episode-1"
    model._lock = threading.RLock()
    return model


def _observation() -> dict:
    return {
        "episode_id": "episode-1",
        "step": 3,
        "instruction": "fold the towel",
        "images": {
            camera: np.zeros((4, 5, 3), dtype=np.uint8) for camera in CAMERA_KEYS
        },
        "state": {
            "follow1_pos": np.zeros(7, dtype=np.float32),
            "follow2_pos": np.ones(7, dtype=np.float32),
        },
    }


def test_observation_requires_exact_public_joint_contract() -> None:
    model = _model_shell()
    observation = _observation()
    model.ingest_observation(observation)
    assert model._latest_obs["instruction"] == "fold the towel"
    assert model._latest_obs["state"]["follow2_pos"].shape == (7,)
    observation["state"]["follow2_pos"][0] = 8
    assert model._latest_obs["state"]["follow2_pos"][0] == 1


@pytest.mark.parametrize(
    ("mutator", "error"),
    [
        (lambda obs: obs["images"].pop("camera_left"), "missing camera"),
        (
            lambda obs: obs["images"].__setitem__(
                "camera_front", np.zeros((4, 5, 3), dtype=np.float32)
            ),
            "dtype uint8",
        ),
        (lambda obs: obs["state"].pop("follow2_pos"), "missing state"),
        (
            lambda obs: obs["state"].__setitem__(
                "follow1_pos", np.zeros(8, dtype=np.float32)
            ),
            "shape",
        ),
        (lambda obs: obs.__setitem__("instruction", ""), "current instruction"),
    ],
)
def test_invalid_observation_fails_before_model(mutator, error: str) -> None:
    model = _model_shell()
    observation = _observation()
    mutator(observation)
    with pytest.raises((KeyError, TypeError, ValueError), match=error):
        model.ingest_observation(observation)


def test_physical_h32x14_packs_to_exact_h32x26() -> None:
    physical = np.arange(ACTION_HORIZON * 14, dtype=np.float32).reshape(
        ACTION_HORIZON, 14
    )
    packed = pack_x2real_action(physical)
    assert packed.shape == (32, 26)
    assert packed.dtype == np.float32
    np.testing.assert_array_equal(packed[:, :14], physical)
    np.testing.assert_array_equal(packed[:, 14:], 0.0)

    with pytest.raises(ValueError, match="must be"):
        pack_x2real_action(np.zeros((31, 14), dtype=np.float32))


def test_deployment_declares_x2robot_v2_h32_contract() -> None:
    definition = load_policy_definition("fastwam_x2robot")
    assert definition.metadata["action_horizon"] == 32
    assert definition.metadata["action_dim"] == 26
    assert definition.metadata["physical_action_dim"] == 14
    pair = definition.metadata["supported_schema_pairs"][0]
    assert pair == {
        "observation_schema": "bimanual_rgb_joint@1",
        "action_schema": "bimanual_joint_position_26d@1",
        "embodiment_constraints": ["ex001_6r@1"],
    }
    assert definition.metadata["session_mode"] == "exclusive"
