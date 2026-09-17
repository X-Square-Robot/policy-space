from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "test_arx_policy", ROOT / "policies/openpi/pi05_arx_r5_eef/model.py"
)
assert SPEC and SPEC.loader
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def observation(instruction="pick up the block"):
    return {
        "images": {
            "camera_front": np.full((8, 9, 3), 11, dtype=np.uint8),
            "camera_left": np.full((8, 9, 3), 22, dtype=np.uint8),
            "camera_right": np.full((8, 9, 3), 33, dtype=np.uint8),
        },
        "state": {
            "follow1_pos": np.array([.1, .2, .3, .4, .5, .6, .7], np.float32),
            "follow2_pos": np.array([-.1, -.2, -.3, -.4, -.5, -.6, .2], np.float32),
        },
        "instruction": instruction,
    }


def test_preserves_pose_units_arm_order_and_three_camera_roles():
    obs = observation()
    native = M.build_arx_observation(obs)
    np.testing.assert_array_equal(
        native["observation.state"],
        np.concatenate([obs["state"]["follow1_pos"], obs["state"]["follow2_pos"]]),
    )
    transformed = M.ArxInputs()(native)
    assert list(transformed["image"]) == ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
    assert [int(image[0, 0, 0]) for image in transformed["image"].values()] == [11, 22, 33]
    assert all(transformed["image_mask"].values())
    assert transformed["prompt"] == "pick up the block"
    obs["state"]["follow1_pos"][:] = 0
    obs["images"]["camera_right"][:] = 0
    assert native["observation.state"][0] != 0
    assert transformed["image"]["right_wrist_0_rgb"][0, 0, 0] == 33


@pytest.mark.parametrize("field", ["follow1_pos", "follow2_pos"])
def test_rejects_joint_or_rot6d_width_and_nonfinite_state(field):
    obs = observation()
    obs["state"][field] = np.zeros(10, np.float32)
    with pytest.raises(ValueError, match="finite"):
        M.build_arx_observation(obs)
    obs["state"][field] = np.zeros(7, np.float32)
    obs["state"][field][0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        M.build_arx_observation(obs)


def test_rejects_unnormalized_grippers_and_missing_images():
    obs = observation()
    obs["state"]["follow1_pos"][-1] = 4.5
    with pytest.raises(ValueError, match="normalized"):
        M.build_arx_observation(obs)
    obs = observation()
    del obs["images"]["camera_right"]
    with pytest.raises(ValueError, match="camera_right"):
        M.build_arx_observation(obs)
    obs = observation()
    obs["images"]["camera_front"] = np.zeros((2, 2, 3), np.float32)
    with pytest.raises(ValueError, match="uint8"):
        M.build_arx_observation(obs)


def test_action_output_drops_padding_without_pose_conversion_or_state_echo():
    padded = np.arange(32 * 32, dtype=np.float32).reshape(32, 32) / 100
    padded[:, 6] = -.2
    padded[:, 13] = 1.2
    wire = M.validate_arx_actions(M.ArxOutputs()({"actions": padded}))
    assert wire.shape == (32, 14)
    pose_dims = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    np.testing.assert_array_equal(wire[:, pose_dims], padded[:, pose_dims])
    np.testing.assert_array_equal(wire[:, 6], 0)
    np.testing.assert_array_equal(wire[:, 13], 1)


@pytest.mark.parametrize("shape", [(31, 14), (33, 14), (32, 20), (32, 32), (14,)])
def test_rejects_wrong_action_width_or_horizon(shape):
    with pytest.raises(RuntimeError, match="32, 14"):
        M.validate_arx_actions({"actions": np.zeros(shape, np.float32)})


def test_rejects_nonfinite_actions():
    actions = np.zeros((32, 14), np.float32)
    actions[0, 0] = np.inf
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        M.validate_arx_actions({"actions": actions})


def model_with_fake_backend():
    model = M.Model.__new__(M.Model)
    model._default_instruction = ""
    model._instruction = ""
    model._latest_obs = None
    class Backend:
        def infer(self, inputs):
            assert inputs["observation.state"].shape == (14,)
            return {"actions": np.tile(inputs["observation.state"], (32, 1))}
    model._policy = Backend()
    return model


def test_episode_instruction_and_observation_do_not_leak():
    model = model_with_fake_backend()
    model.initialize_episode({"instruction": "episode one"})
    model.ingest_observation(observation(""))
    assert model._latest_obs["prompt"] == "episode one"
    assert model.infer_actions()["actions"].shape == (32, 14)
    model.finalize_episode()
    with pytest.raises(RuntimeError, match="buffered"):
        model.infer_actions()
    model.initialize_episode({})
    with pytest.raises(ValueError, match="instruction"):
        model.ingest_observation(observation(""))
    assert model._latest_obs is None


def test_invalid_ingest_cannot_reuse_previous_observation():
    model = model_with_fake_backend()
    model.ingest_observation(observation())
    with pytest.raises(TypeError):
        model.ingest_observation({})
    with pytest.raises(RuntimeError, match="buffered"):
        model.infer_actions()


def checkpoint_fixture(tmp_path):
    (tmp_path / "params").mkdir()
    (tmp_path / "params/_METADATA").write_text("{}")
    (tmp_path / "params/manifest.ocdbt").write_bytes(b"manifest")
    (tmp_path / "_CHECKPOINT_METADATA").write_text(json.dumps({"commit_timestamp_nsecs": 1}))
    assets = tmp_path / "assets/pi05_arx_ee"
    assets.mkdir(parents=True)
    stats = {field: {key: [0.] * 14 for key in ["mean", "std", "q01", "q99"]}
             for field in ["state", "actions"]}
    (assets / "norm_stats.json").write_text(json.dumps({"norm_stats": stats}))
    return tmp_path


def test_checkpoint_requires_committed_write_and_matching_norm_width(tmp_path):
    checkpoint = checkpoint_fixture(tmp_path)
    M._validate_checkpoint(checkpoint)
    (checkpoint / "_CHECKPOINT_METADATA").write_text("{}")
    with pytest.raises(ValueError, match="commit"):
        M._validate_checkpoint(checkpoint)
    (checkpoint / "_CHECKPOINT_METADATA").write_text(json.dumps({"commit_timestamp_nsecs": 1}))
    f = checkpoint / "assets/pi05_arx_ee/norm_stats.json"
    data = json.loads(f.read_text())
    data["norm_stats"]["state"]["q01"] = [0.] * 20
    f.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="14"):
        M._validate_checkpoint(checkpoint)


def test_deployment_negotiates_arx_not_ex001_and_checks_full_lifecycle():
    from policy_space.server import load_policy_definition
    from policy_space.runtime.server import PolicySpaceServer
    definition = load_policy_definition("openpi_pi05_arx_ee")
    metadata = definition.metadata
    assert metadata["action_horizon"] == 32
    assert metadata["action_dim"] == 14
    assert metadata["model_action_dim"] == 32
    assert metadata["gripper_convention"] == "normalized_01"
    assert metadata["action_fps"] == 20
    pair = metadata["supported_schema_pairs"][0]
    assert pair["embodiment_constraints"] == ["arx_r5@1"]
    server = PolicySpaceServer(model_with_fake_backend(), metadata=metadata)
    selection = {
        "episode_id": "arx-test",
        "observation_schema": "bimanual_rgb_ee@1",
        "action_schema": "bimanual_ee_absolute@1",
        "embodiment_schema": "arx_r5@1",
    }
    with pytest.raises(ValueError, match="compatible"):
        server._dispatch("initialize_episode", {**selection, "embodiment_schema": "ex001_6r@1"})
    server._dispatch("initialize_episode", selection)
    response = server._dispatch("infer_actions", {**observation(), "episode_id": "arx-test"})
    assert response["actions"].shape == (32, 14)
    server._dispatch("finalize_episode", {"episode_id": "arx-test"})


def test_runtime_metadata_respects_static_deployment_contract():
    from policy_space.server import load_policy_definition
    metadata = load_policy_definition("openpi_pi05_arx_ee").metadata
    model = model_with_fake_backend()
    model.checkpoint_path = Path("/checkpoint/latest")
    dynamic = model.runtime_metadata()
    assert not (dynamic.keys() & metadata.keys())
    assert dynamic["openpi_wire_action_shape"] == [32, 14]
