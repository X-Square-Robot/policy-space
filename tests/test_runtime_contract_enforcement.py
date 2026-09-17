"""Runtime schema negotiation and action-boundary tests."""

from __future__ import annotations

import numpy as np
import pytest
from msgpack import Packer

from policy_space import PolicyModel
from policy_space.runtime.server import PolicySpaceServer, _safe_packb, _safe_unpackb


class _Model(PolicyModel):
    def __init__(self, actions: np.ndarray) -> None:
        super().__init__({})
        self.actions = actions
        self.initialized = 0

    def initialize_episode(self, episode_info=None) -> None:
        self.initialized += 1

    def ingest_observation(self, obs) -> None:
        pass

    def infer_actions(self) -> np.ndarray:
        return self.actions


def _metadata() -> dict:
    return {
        "session_mode": "exclusive",
        "supported_schema_pairs": [
            {
                "observation_schema": "bimanual_rgb_ee@1",
                "action_schema": "bimanual_ee_absolute@1",
                "embodiment_constraints": ["ex001_6r@1"],
            }
        ],
    }


def _selection() -> dict:
    return {
        "episode_id": "episode-1",
        "observation_schema": "bimanual_rgb_ee@1",
        "action_schema": "bimanual_ee_absolute@1",
        "embodiment_schema": "ex001_6r@1",
    }


def _observation() -> dict:
    return {
        "episode_id": "episode-1",
        "step": 0,
        "instruction": "hold",
        "images": {
            "camera_front": np.zeros((2, 2, 3), dtype=np.uint8),
            "camera_left": np.zeros((2, 2, 3), dtype=np.uint8),
            "camera_right": np.zeros((2, 2, 3), dtype=np.uint8),
        },
        "state": {
            "follow1_pos": np.zeros(7, dtype=np.float32),
            "follow2_pos": np.zeros(7, dtype=np.float32),
        },
    }


def test_schema_mismatch_fails_before_model_initialization() -> None:
    model = _Model(np.zeros((1, 14), dtype=np.float32))
    server = PolicySpaceServer(model, metadata=_metadata())
    request = {**_selection(), "action_schema": "bimanual_joint_position@1"}

    with pytest.raises(ValueError, match="No compatible"):
        server._dispatch("initialize_episode", request)

    assert model.initialized == 0


def test_invalid_action_width_is_rejected_before_serialization() -> None:
    server = PolicySpaceServer(
        _Model(np.zeros((1, 13), dtype=np.float32)), metadata=_metadata()
    )
    server._dispatch("initialize_episode", _selection())

    with pytest.raises(ValueError, match=r"\[T,14\]"):
        server._dispatch("infer_actions", _observation())


def test_action_horizon_mismatch_is_rejected_before_serialization() -> None:
    metadata = {**_metadata(), "action_horizon": 2}
    server = PolicySpaceServer(
        _Model(np.zeros((1, 14), dtype=np.float32)), metadata=metadata
    )
    server._dispatch("initialize_episode", _selection())

    with pytest.raises(ValueError, match="action_horizon 2; got 1"):
        server._dispatch("infer_actions", _observation())


def test_missing_required_observation_field_is_rejected_before_model() -> None:
    server = PolicySpaceServer(
        _Model(np.zeros((1, 14), dtype=np.float32)), metadata=_metadata()
    )
    server._dispatch("initialize_episode", _selection())
    observation = _observation()
    del observation["state"]["follow2_pos"]

    with pytest.raises(ValueError, match="follow2_pos"):
        server._dispatch("infer_actions", observation)


def test_missing_required_camera_role_is_rejected_before_model() -> None:
    server = PolicySpaceServer(
        _Model(np.zeros((1, 14), dtype=np.float32)), metadata=_metadata()
    )
    server._dispatch("initialize_episode", _selection())
    observation = _observation()
    del observation["images"]["camera_right"]

    with pytest.raises(ValueError, match="camera_right"):
        server._dispatch("infer_actions", observation)


def test_msgpack_decoder_accepts_legacy_integer_environment_keys() -> None:
    payload = {"endpoint": "finalize_episode", "env_steps": {0: 12, 1: 9}}

    assert _safe_unpackb(_safe_packb(payload)) == payload


def test_msgpack_decoder_limits_integer_keys_to_lifecycle_environment_maps() -> None:
    payload = {"endpoint": "infer_actions", "unexpected": {0: 12}}

    with pytest.raises(ValueError, match="integer map keys"):
        _safe_unpackb(_safe_packb(payload))


def test_msgpack_decoder_rejects_non_protocol_map_key_types() -> None:
    payload = {"endpoint": "finalize_episode", "env_steps": {1.5: 12}}

    with pytest.raises(ValueError, match="map key"):
        _safe_unpackb(_safe_packb(payload))


def test_msgpack_decoder_rejects_keys_before_numeric_collision() -> None:
    packer = Packer(use_bin_type=True)
    payload = b"".join(
        [
            packer.pack_map_header(1),
            packer.pack("env_steps"),
            packer.pack_map_header(2),
            packer.pack(1),
            packer.pack(12),
            packer.pack(1.0),
            packer.pack(9),
        ]
    )

    with pytest.raises(ValueError, match="map key"):
        _safe_unpackb(payload)


def test_msgpack_decoder_rejects_duplicate_protocol_keys() -> None:
    packer = Packer(use_bin_type=True)
    payload = b"".join(
        [
            packer.pack_map_header(2),
            packer.pack("endpoint"),
            packer.pack("infer_actions"),
            packer.pack("endpoint"),
            packer.pack("finalize_episode"),
        ]
    )

    with pytest.raises(ValueError, match="duplicate map key"):
        _safe_unpackb(payload)


def test_payload_depth_limit_is_enforced() -> None:
    server = PolicySpaceServer(
        _Model(np.zeros((1, 14), dtype=np.float32)),
        metadata=_metadata(),
        max_payload_depth=2,
    )

    with pytest.raises(ValueError, match="nesting depth"):
        server._validate_payload_depth({"a": {"b": {"c": 1}}})


def test_request_rate_limit_is_enforced_per_connection() -> None:
    server = PolicySpaceServer(
        _Model(np.zeros((1, 14), dtype=np.float32)),
        metadata=_metadata(),
        max_requests_per_minute=1,
    )
    server._check_request_rate()
    with pytest.raises(RuntimeError, match="request rate limit"):
        server._check_request_rate()


def test_multiplexed_session_limit_rejects_a_second_active_episode() -> None:
    metadata = {**_metadata(), "session_mode": "multiplexed_serial"}
    server = PolicySpaceServer(
        _Model(np.zeros((1, 14), dtype=np.float32)),
        metadata=metadata,
        max_sessions=1,
    )
    first_owner = object()
    server._claim_multiplexed_session(_selection(), first_owner)

    with pytest.raises(RuntimeError, match="session limit"):
        server._claim_multiplexed_session(
            {**_selection(), "episode_id": "episode-2"}, object()
        )


def test_connection_limit_returns_retryable_handshake_metadata() -> None:
    class Connection:
        def __init__(self) -> None:
            self.responses: list[object] = []
            self.closed: tuple[int, str] | None = None

        def send(self, value: object) -> None:
            self.responses.append(value)

        def close(self, code: int, reason: str) -> None:
            self.closed = (code, reason)

    server = PolicySpaceServer(
        _Model(np.zeros((1, 14), dtype=np.float32)),
        metadata=_metadata(),
        max_connections=1,
    )
    assert server._connection_slots.acquire(blocking=False)
    connection = Connection()
    try:
        server._handle_client(connection)
    finally:
        server._connection_slots.release()

    metadata = _safe_unpackb(connection.responses[0])
    assert metadata["error"] == "connection_limit"
    assert metadata["retryable"] is True
    assert connection.closed is not None and connection.closed[0] == 1013
