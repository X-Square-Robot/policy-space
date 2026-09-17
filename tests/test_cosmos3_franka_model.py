"""Cosmos3 Franka bundle contract tests without loading the GPU model."""

from __future__ import annotations

import numpy as np
import pytest

from policies.cosmos3.cosmos3_franka_joint.model import CAMERA_MAP, Model
from policy_space.server import load_policy_definition


def _model_shell() -> Model:
    model = Model.__new__(Model)
    model._camera_map = dict(CAMERA_MAP)
    model._instruction = ""
    model._require_instruction = True
    model._model_action_horizon = 32
    model._action_horizon = 24
    model._latest_obs = None
    return model


def _observation() -> dict:
    return {
        "images": {
            camera: np.zeros((4, 5, 3), dtype=np.uint8) for camera in CAMERA_MAP
        },
        "state": {
            "joint_position": np.zeros(7, dtype=np.float32),
            "gripper_position": np.zeros(1, dtype=np.float32),
        },
        "instruction": "pick up the block",
    }


def test_observation_mapping_is_exact_and_copied() -> None:
    model = _model_shell()
    observation = _observation()
    model.ingest_observation(observation)
    request = model._build_request(model._latest_obs)

    assert request["prompt"] == "pick up the block"
    assert set(request) == {
        *CAMERA_MAP.values(),
        "observation/joint_position",
        "observation/gripper_position",
        "prompt",
    }
    observation["images"]["wrist"][0, 0, 0] = 255
    assert request[CAMERA_MAP["wrist"]][0, 0, 0] == 0


@pytest.mark.parametrize(
    ("mutator", "error"),
    [
        (lambda obs: obs["images"].pop("exterior_2"), "missing camera"),
        (
            lambda obs: obs["images"].__setitem__(
                "wrist", np.zeros((4, 5, 3), dtype=np.float32)
            ),
            "dtype uint8",
        ),
        (lambda obs: obs["state"].pop("gripper_position"), "missing state"),
        (
            lambda obs: obs["state"].__setitem__(
                "joint_position", np.zeros(8, dtype=np.float32)
            ),
            "shape",
        ),
        (lambda obs: obs.__setitem__("instruction", ""), "current instruction"),
    ],
)
def test_invalid_observation_fails_before_inference(mutator, error: str) -> None:
    model = _model_shell()
    observation = _observation()
    mutator(observation)
    with pytest.raises((KeyError, TypeError, ValueError), match=error):
        model.ingest_observation(observation)


def test_model_h32_is_truncated_to_exact_wire_h24() -> None:
    model = _model_shell()
    source = np.arange(32 * 8, dtype=np.float32).reshape(32, 8)
    result = model._parse_actions({"action": source})
    assert result.shape == (24, 8)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, source[:24])

    with pytest.raises(RuntimeError, match="model output"):
        model._parse_actions({"action": np.zeros((24, 8), dtype=np.float32)})


def test_deployment_declares_franka_v2_h24_contract() -> None:
    definition = load_policy_definition("cosmos3_franka")
    assert definition.metadata["action_horizon"] == 24
    assert definition.metadata["action_dim"] == 8
    pair = definition.metadata["supported_schema_pairs"][0]
    assert pair == {
        "observation_schema": "single_arm_rgb_joint@1",
        "action_schema": "single_arm_joint_position@1",
        "embodiment_constraints": ["franka@1"],
    }
    assert definition.metadata["session_mode"] == "exclusive"
