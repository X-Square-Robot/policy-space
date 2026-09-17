"""PolicySpace deploy.yaml loading contracts."""

from pathlib import Path

import pytest

import policy_space.runtime.server as server_module
from policy_space.protocol import (
    SESSION_MODE_EXCLUSIVE,
    SESSION_MODE_MULTIPLEXED_SERIAL,
    normalize_session_mode,
)
from policy_space.server import load_policy_definition, load_policy_model

ROOT = Path(__file__).resolve().parents[1]


def test_server_disables_websocket_keepalive_for_long_action_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, metadata = load_policy_model("example_policy")
    captured: dict[str, object] = {}

    class _FakeWebSocketServer:
        def serve_forever(self) -> None:
            return None

    class _FakeServeContext:
        def __enter__(self) -> _FakeWebSocketServer:
            return _FakeWebSocketServer()

        def __exit__(self, *_args: object) -> None:
            return None

    def fake_serve(*args: object, **kwargs: object) -> _FakeServeContext:
        captured["args"] = args
        captured.update(kwargs)
        return _FakeServeContext()

    monkeypatch.setattr(server_module, "serve", fake_serve)

    server_module.PolicySpaceServer(model, metadata=metadata).serve_forever()

    assert captured["ping_interval"] is None
    assert captured["ping_timeout"] is None


def test_load_policy_definition_uses_direct_model_cfg_and_metadata() -> None:
    definition = load_policy_definition("dreamzero_x2robot")

    assert (
        definition.model_path
        == ROOT / "policies" / "dreamzero" / "dreamzero_artixon_arm_6a_joint" / "model.py"
    )
    assert definition.model_class_name == "Model"
    assert definition.model_cfg["history_window"] == 25
    assert definition.model_cfg["action_horizon"] == 24
    assert definition.model_cfg["image_resolution"] == [176, 320]
    assert definition.model_cfg["inference_method"] == "lazy_joint_forward_causal"
    assert definition.model_cfg["replan_context_mode"] == "causal"
    assert definition.model_cfg["execute_horizon"] == 24
    assert definition.model_cfg["save_video_pred"] is False
    assert (
        definition.model_cfg["video_output_dir"] == "/tmp/dreamzero_x2robot_video_pred"
    )
    assert definition.model_cfg["video_fps"] == 15
    assert definition.model_cfg["gripper_debounce"]["enabled"] is False
    assert definition.model_cfg["gripper_debounce"]["min_stable_steps"] == 3
    assert definition.model_cfg["gripper_debounce"]["mode"] == "continuous_slew"
    assert definition.model_cfg["gripper_debounce"]["closed_target"] == 0.0
    assert definition.model_cfg["gripper_debounce"]["open_target"] == 1.0
    assert definition.model_cfg["gripper_debounce"]["max_delta_per_step"] == 0.06
    assert definition.model_cfg["gripper_debounce"]["deadband"] == 0.02
    assert definition.model_cfg["gripper_debounce"]["ema_alpha"] == 0.65
    assert definition.model_cfg["gripper_debounce"]["max_close_delta_per_step"] == 0.20
    assert definition.model_cfg["gripper_debounce"]["max_open_delta_per_step"] == 0.10
    assert definition.model_cfg["gripper_debounce"]["release_threshold"] == 0.75
    assert definition.model_cfg["gripper_debounce"]["release_stable_steps"] == 4
    assert definition.model_cfg["action_smoothing"] == {
        "enabled": True,
        "interpolation_multiplier": 2,
        "window_length": 7,
        "polyorder": 2,
        "boundary_blend_steps": 6,
    }
    assert "backend_port" not in definition.model_cfg
    assert definition.metadata["policy_name"] == "dreamzero_x2robot"
    assert definition.metadata["action_dim"] == 26
    assert definition.metadata["action_mode"] == "joint_position"
    assert definition.metadata["action_horizon"] == 24
    assert "recommended_execute_horizon" not in definition.metadata
    assert definition.metadata["recommended_connect_chunks"] is False
    assert definition.metadata["recommended_ingest_observation_each_step"] is True
    assert definition.metadata["session_mode"] == SESSION_MODE_EXCLUSIVE


def test_session_mode_validation_is_fail_closed() -> None:
    assert normalize_session_mode(None) == SESSION_MODE_EXCLUSIVE
    assert (
        normalize_session_mode(SESSION_MODE_MULTIPLEXED_SERIAL)
        == SESSION_MODE_MULTIPLEXED_SERIAL
    )
    with pytest.raises(ValueError, match="session_mode"):
        normalize_session_mode("parallel")


@pytest.mark.parametrize(
    ("policy_name", "expected"),
    [
        ("wallx/wall_oss_05_artixon_arm_6a_eef", SESSION_MODE_MULTIPLEXED_SERIAL),
        ("openpi_x2robot", SESSION_MODE_MULTIPLEXED_SERIAL),
        ("dreamzero_x2robot", SESSION_MODE_EXCLUSIVE),
        ("dreamzero_droid", SESSION_MODE_EXCLUSIVE),
    ],
)
def test_shipped_policy_session_modes_are_explicit_and_safe(
    policy_name: str, expected: str
) -> None:
    assert load_policy_definition(policy_name).metadata["session_mode"] == expected


def test_load_policy_model_still_constructs_lightweight_policy() -> None:
    model, metadata = load_policy_model("example_policy")

    assert model.cfg == {"action_dim": 14, "action_horizon": 32, "gain": 0.0}
    assert model.action_dim == 14
    assert metadata == {
        "supported_schema_pairs": [
            {
                "observation_schema": "bimanual_rgb_ee@1",
                "action_schema": "bimanual_ee_absolute@1",
                "embodiment_constraints": ["ex001_6r@1"],
            }
        ],
        "protocol": "policy_space",
        "protocol_version": "2",
        "embodiment": "ArtiXon Arm-6A",
        "action_mode": "ik_eef_abs",
        "action_dim": 14,
        "action_horizon": 32,
        "action_type": "absolute_ee",
        "policy_name": "example_policy",
        "session_mode": "multiplexed_serial",
    }


@pytest.mark.parametrize(
    ("policy_name", "template_name"),
    [
        ("example_policy", "bimanual_ee"),
        ("example_franka_policy", "franka_joint"),
        ("example_mobile_policy", "mobile_manipulation"),
    ],
)
def test_template_catalog_preserves_name_and_legacy_path_resolution(
    policy_name: str, template_name: str
) -> None:
    expected_dir = ROOT / "policies" / "templates" / template_name
    requests = [
        policy_name,
        ROOT / "examples" / policy_name,
        ROOT / "examples" / policy_name / "deploy.yaml",
        ROOT / "policies" / policy_name,
        ROOT / "policies" / policy_name / "deploy.yaml",
        expected_dir,
        expected_dir / "deploy.yaml",
    ]

    for request in requests:
        definition = load_policy_definition(request)
        assert definition.model_path.parent == expected_dir
