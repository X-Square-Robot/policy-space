from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODEL_PATH = (
    Path(__file__).parents[1]
    / "policies"
    / "openpi"
    / "pi05_artixon_arm_6a_joint"
    / "model.py"
)
SPEC = importlib.util.spec_from_file_location("pi05_joint_model_test", MODEL_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_build_joint_observation_preserves_14d_state_and_camera_contract() -> None:
    obs = {
        "images": {
            "camera_front": np.zeros((4, 5, 3), dtype=np.uint8),
            "camera_left": np.ones((4, 5, 3), dtype=np.uint8),
            "camera_right": np.full((4, 5, 3), 2, dtype=np.uint8),
        },
        "state": {
            "follow1_pos": np.arange(7, dtype=np.float32),
            "follow2_pos": np.arange(7, 14, dtype=np.float32),
        },
        "instruction": "pick up the block",
    }

    result = MODULE.build_joint_observation(obs, instruction="")

    np.testing.assert_array_equal(result["observation/state"], np.arange(14, dtype=np.float32))
    assert result["observation/image"].shape == (4, 5, 3)
    assert result["observation/left_wrist_image"].shape == (4, 5, 3)
    assert result["observation/right_wrist_image"].shape == (4, 5, 3)
    assert result["prompt"] == "pick up the block"


def test_build_joint_observation_rejects_non_7d_arm_state() -> None:
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    obs = {
        "images": {"camera_front": image, "camera_left": image, "camera_right": image},
        "state": {"follow1_pos": [0] * 6, "follow2_pos": [0] * 7},
    }

    with pytest.raises(ValueError, match="follow1_pos.*7"):
        MODULE.build_joint_observation(obs, instruction="task")


def test_validate_joint_actions_keeps_14d_absolute_output() -> None:
    actions = np.arange(28, dtype=np.float32).reshape(2, 14)

    result = MODULE.validate_joint_actions({"actions": actions})

    np.testing.assert_array_equal(result, actions)


def test_validate_joint_actions_rejects_model_padding_or_nan() -> None:
    with pytest.raises(RuntimeError, match="14"):
        MODULE.validate_joint_actions({"actions": np.zeros((2, 32), dtype=np.float32)})
    with pytest.raises(RuntimeError, match="NaN"):
        bad = np.zeros((2, 14), dtype=np.float32)
        bad[0, 0] = np.nan
        MODULE.validate_joint_actions({"actions": bad})


def test_joint_deploy_declares_the_14d_joint_contract() -> None:
    from policy_space.base import PolicyModel
    from policy_space.server import load_policy_definition

    definition = load_policy_definition("openpi_pi05_x2real_joint")

    assert definition.metadata["action_mode"] == "joint_position"
    assert definition.metadata["action_dim"] == 14
    assert definition.metadata["action_type"] == "absolute_joint"
    assert definition.metadata["recommended_ingest_observation_each_step"] is False
    assert definition.model_cfg["train_config"] == "pi05_x2real_split280"
    assert issubclass(MODULE.Model, PolicyModel)


def test_finalize_episode_discards_buffered_observation() -> None:
    model = MODULE.Model.__new__(MODULE.Model)
    model._latest_obs = {"state": "stale"}

    model.finalize_episode()

    assert model._latest_obs is None
