"""Regression guards for the OpenPI π0.5 ArtiXon Arm-6A integration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


class _FakeOpenPIPolicy:
    def __init__(self, actions: np.ndarray) -> None:
        self.actions = actions
        self.last_observation: dict[str, Any] | None = None

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        self.last_observation = observation
        return {"actions": self.actions}


def _make_model(monkeypatch: pytest.MonkeyPatch, tmp_path, *, action_horizon: int = 4):
    from policies.openpi.pi05_artixon_arm_6a_eef import model as openpi_model

    monkeypatch.delenv("OPENPI_TRAIN_CONFIG", raising=False)
    monkeypatch.delenv("OPENPI_CHECKPOINT_PATH", raising=False)
    monkeypatch.delenv("OPENPI_DEVICE", raising=False)
    monkeypatch.delenv("OPENPI_PYTORCH_COMPILE_MODE", raising=False)
    monkeypatch.delenv("OPENPI_ROTATION_LAYOUT", raising=False)

    predicted_wire = np.array(
        [
            [
                0.10 + step * 0.01,
                -0.20,
                0.30,
                0.05,
                -0.10,
                0.15,
                0.25,
                -0.10,
                0.20 + step * 0.01,
                0.35,
                -0.05,
                0.10,
                -0.15,
                0.75,
            ]
            for step in range(6)
        ],
        dtype=np.float32,
    )
    fake_policy = _FakeOpenPIPolicy(openpi_model._wire_to_model_pose(predicted_wire))
    loaded = {}

    def _fake_create(
        train_config_name: str,
        checkpoint_path: str,
        device: str,
        pytorch_compile_mode: str | None,
        openpi_root: Path,
    ):
        loaded.update(
            train_config_name=train_config_name,
            checkpoint_path=checkpoint_path,
            device=device,
            pytorch_compile_mode=pytorch_compile_mode,
            openpi_root=openpi_root,
        )
        return fake_policy

    monkeypatch.setattr(openpi_model, "_create_openpi_policy", _fake_create)
    model = openpi_model.Model(
        {
            "openpi_root": str(tmp_path),
            "checkpoint_path": str(tmp_path / "checkpoint"),
            "action_horizon": action_horizon,
            "instruction": "rotate the book",
        }
    )
    return model, fake_policy, predicted_wire, loaded


def _observation() -> dict[str, Any]:
    return {
        "images": {
            "camera_front": np.full((4, 5, 3), 0.5, dtype=np.float32),
            "camera_left": np.full((4, 5, 3), 64, dtype=np.uint8),
            "camera_right": np.full((4, 5, 3), 128, dtype=np.uint8),
        },
        "state": {
            "follow1_pos": np.array(
                [0.1, -0.2, 0.3, 0.05, -0.1, 0.15, 0.25], dtype=np.float32
            ),
            "follow2_pos": np.array(
                [-0.1, 0.2, 0.35, -0.05, 0.1, -0.15, 0.75], dtype=np.float32
            ),
        },
        "instruction": "pick up the spoon",
    }


def test_openpi_pi05_artixon_arm_6a_model_builds_openpi_obs_and_preserves_predictions(
    monkeypatch, tmp_path
) -> None:
    from policies.openpi.pi05_artixon_arm_6a_eef import model as openpi_model

    model, fake_policy, predicted_wire, loaded = _make_model(monkeypatch, tmp_path)

    model.ingest_observation(_observation())
    actions = model.infer_actions()

    assert loaded["train_config_name"] == "pi05_x2real_lerobotv2_finetune"
    assert loaded["device"] == "cuda:0"
    assert loaded["pytorch_compile_mode"] is None
    assert model.rotation_layout == "row"
    assert actions.shape == (5, 14)
    current_wire = np.concatenate(
        [_observation()["state"]["follow1_pos"], _observation()["state"]["follow2_pos"]]
    )
    np.testing.assert_array_equal(actions[0], current_wire)
    np.testing.assert_allclose(actions[1:], predicted_wire[:4], rtol=0.0, atol=1e-6)

    policy_obs = fake_policy.last_observation
    assert policy_obs is not None
    assert policy_obs["prompt"] == "pick up the spoon"
    assert policy_obs["state"].shape == (20,)
    np.testing.assert_allclose(
        policy_obs["state"],
        openpi_model._wire_to_model_pose(current_wire),
        rtol=0.0,
        atol=1e-6,
    )
    assert set(policy_obs["images"]) == {
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    }
    assert all(image.shape == (3, 4, 5) for image in policy_obs["images"].values())
    assert all(image.dtype == np.uint8 for image in policy_obs["images"].values())
    assert np.all(policy_obs["images"]["cam_high"] == 127)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda obs: obs["images"].pop("camera_left"), "Missing required cameras"),
        (lambda obs: obs["state"].pop("follow2_pos"), "Missing required state field"),
        (
            lambda obs: obs.update(instruction=""),
            "requires a non-empty task instruction",
        ),
    ],
)
def test_openpi_pi05_artixon_arm_6a_model_rejects_incomplete_observations(
    monkeypatch,
    tmp_path,
    mutation,
    message: str,
) -> None:
    model, _fake_policy, _predicted, _loaded = _make_model(monkeypatch, tmp_path)
    model.instruction = ""
    observation = _observation()
    mutation(observation)
    model.ingest_observation(observation)

    with pytest.raises(ValueError, match=message):
        model.infer_actions()


def test_openpi_pi05_artixon_arm_6a_model_rejects_invalid_action_shape(
    monkeypatch, tmp_path
) -> None:
    model, fake_policy, _predicted, _loaded = _make_model(monkeypatch, tmp_path)
    fake_policy.actions = np.zeros((4, 19), dtype=np.float32)
    model.ingest_observation(_observation())

    with pytest.raises(RuntimeError, match=r"shape \(T, 20\)"):
        model.infer_actions()


def test_openpi_pi05_artixon_arm_6a_pose_projection_round_trips_wire_contract() -> None:
    from policies.openpi.pi05_artixon_arm_6a_eef import model as openpi_model

    wire = np.array(
        [
            [0.1, 0.2, 0.3, 0.2, -0.3, 0.4, 0.1, -0.1, -0.2, 0.4, -0.2, 0.3, -0.4, 0.9],
            [0.2, 0.1, 0.4, -0.4, 0.3, -0.2, 0.2, -0.2, -0.1, 0.3, 0.4, -0.3, 0.2, 0.8],
        ],
        dtype=np.float32,
    )

    model_pose = openpi_model._wire_to_model_pose(wire)
    assert model_pose.shape == (2, 20)
    np.testing.assert_allclose(
        openpi_model._model_to_wire_pose(model_pose),
        wire,
        rtol=0.0,
        atol=1e-6,
    )


def test_openpi_pi05_artixon_arm_6a_replaces_only_aloha14_output_transform() -> None:
    from policies.openpi.pi05_artixon_arm_6a_eef import model as openpi_model

    class _FakeAlohaOutputs:
        def __call__(self, data):
            return {"actions": data["actions"][:, :14]}

    class _Composite:
        def __init__(self, transforms):
            self.transforms = transforms

        def __call__(self, data):
            for transform in self.transforms:
                data = transform(data)
            return data

    policy = SimpleNamespace(_output_transform=_Composite((_FakeAlohaOutputs(),)))
    openpi_model._restore_x2_output_transform(policy, _FakeAlohaOutputs, _Composite)

    result = policy._output_transform({"actions": np.ones((3, 32), dtype=np.float32)})
    assert result["actions"].shape == (3, 20)

    with pytest.raises(RuntimeError, match="found 0"):
        openpi_model._restore_x2_output_transform(
            SimpleNamespace(_output_transform=_Composite(())),
            _FakeAlohaOutputs,
            _Composite,
        )


def test_openpi_pi05_artixon_arm_6a_train_config_uses_manifest_compatible_fallback() -> None:
    from policies.openpi.pi05_artixon_arm_6a_eef import model as openpi_model

    sentinel = object()

    class _TrainingConfigs:
        @staticmethod
        def get_config(name):
            if name == openpi_model.COMPATIBLE_TRAIN_CONFIG:
                return sentinel
            raise KeyError(name)

    assert (
        openpi_model._get_openpi_train_config(
            _TrainingConfigs, openpi_model.DEFAULT_TRAIN_CONFIG
        )
        is sentinel
    )
    with pytest.raises(ValueError, match="Unknown OpenPI train_config"):
        openpi_model._get_openpi_train_config(_TrainingConfigs, "not_a_real_config")


def test_openpi_pi05_artixon_arm_6a_rejects_incompatible_train_transform_contract() -> None:
    from policies.openpi.pi05_artixon_arm_6a_eef import model as openpi_model

    valid = SimpleNamespace(
        model=SimpleNamespace(action_horizon=32),
        data=SimpleNamespace(adapt_to_pi=False, use_delta_joint_actions=True),
    )
    openpi_model._validate_openpi_train_config(valid)

    invalid = SimpleNamespace(
        model=SimpleNamespace(action_horizon=10),
        data=SimpleNamespace(adapt_to_pi=True, use_delta_joint_actions=False),
    )
    with pytest.raises(ValueError, match="model.action_horizon"):
        openpi_model._validate_openpi_train_config(invalid)


def test_openpi_pi05_artixon_arm_6a_compile_mode_defaults_off_and_validates_override(
    monkeypatch,
) -> None:
    from policies.openpi.pi05_artixon_arm_6a_eef import model as openpi_model

    monkeypatch.delenv("OPENPI_PYTORCH_COMPILE_MODE", raising=False)
    assert openpi_model._resolve_pytorch_compile_mode({}) is None
    assert (
        openpi_model._resolve_pytorch_compile_mode(
            {"pytorch_compile_mode": "reduce-overhead"}
        )
        == "reduce-overhead"
    )

    monkeypatch.setenv("OPENPI_PYTORCH_COMPILE_MODE", "max-autotune-no-cudagraphs")
    assert (
        openpi_model._resolve_pytorch_compile_mode({"pytorch_compile_mode": None})
        == "max-autotune-no-cudagraphs"
    )

    monkeypatch.setenv("OPENPI_PYTORCH_COMPILE_MODE", "not-a-mode")
    with pytest.raises(ValueError, match="pytorch_compile_mode"):
        openpi_model._resolve_pytorch_compile_mode({})


def test_openpi_pi05_artixon_arm_6a_supports_original_row_and_standard_column_rot6d(
    monkeypatch,
) -> None:
    from policies.openpi.pi05_artixon_arm_6a_eef import model as openpi_model

    euler = np.array([[0.2, -0.3, 0.4], [-0.5, 0.1, -0.2]], dtype=np.float32)
    for layout in ("row", "column"):
        encoded = openpi_model._euler_xyz_to_rot6d(euler, layout=layout)
        decoded = openpi_model._rot6d_to_euler_xyz(encoded, layout=layout)
        np.testing.assert_allclose(decoded, euler, rtol=0.0, atol=1e-6)

    monkeypatch.setenv("OPENPI_ROTATION_LAYOUT", "column")
    assert openpi_model._resolve_rotation_layout({"rotation_layout": "row"}) == "column"
    monkeypatch.setenv("OPENPI_ROTATION_LAYOUT", "invalid")
    with pytest.raises(ValueError, match="rotation_layout"):
        openpi_model._resolve_rotation_layout({})


def test_openpi_pi05_artixon_arm_6a_requires_exact_20d_quantile_stats() -> None:
    from policies.openpi.pi05_artixon_arm_6a_eef import model as openpi_model

    class _Normalize:
        def __init__(self, width):
            self.use_quantiles = True
            values = np.ones(width, dtype=np.float32)
            stats = SimpleNamespace(mean=values, std=values, q01=values, q99=values)
            self.norm_stats = {"state": stats, "actions": stats}

    policy = SimpleNamespace(
        _input_transform=SimpleNamespace(transforms=(_Normalize(20),))
    )
    openpi_model._validate_x2_normalization(policy, _Normalize)

    policy._input_transform.transforms = (_Normalize(14),)
    with pytest.raises(RuntimeError, match=r"state.mean must have shape \(20,\)"):
        openpi_model._validate_x2_normalization(policy, _Normalize)


def test_openpi_pi05_artixon_arm_6a_deploy_contract_matches_the_adapter() -> None:
    deploy_path = (
        ROOT / "policies" / "openpi" / "pi05_artixon_arm_6a_eef" / "deploy.yaml"
    )
    deploy = yaml.safe_load(deploy_path.read_text(encoding="utf-8"))
    metadata = deploy["metadata"]
    model_cfg = deploy["model"]["cfg"]

    assert deploy["policy_name"] == "openpi_x2robot"
    assert metadata["action_mode"] == "ik_eef_abs"
    assert model_cfg["action_horizon"] == 32
    assert metadata["action_horizon"] == 33
    assert metadata["action_dim"] == 14
    assert metadata["model_action_dim"] == 20
    assert metadata["model_rotation"] == "rot6d_row"
    assert metadata["gripper_convention"] == "normalized_01"
    assert metadata["wire_pose_reference"] == "init_ee_pose"
    assert model_cfg["train_config"] == "pi05_x2real_lerobotv2_finetune"
    assert model_cfg["pytorch_compile_mode"] is None
    assert model_cfg["rotation_layout"] == "row"
    assert model_cfg["state_fields"] == ["follow1_pos", "follow2_pos"]
