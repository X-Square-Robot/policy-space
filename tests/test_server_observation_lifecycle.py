"""PolicySpace server observation lifecycle regressions."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import pytest
from websockets.sync.client import connect
from websockets.sync.server import serve

from policy_space.base import PolicyModel
from policy_space.server import (
    PolicySpaceServer,
    _pack_array,
    _safe_packb,
    _safe_unpackb,
    _unpack_array,
)


class _RecordingModel(PolicyModel):
    def __init__(self) -> None:
        super().__init__({})
        self.observations: list[int] = []

    def initialize_episode(self, episode_info=None) -> None:
        self.observations.clear()

    def ingest_observation(self, obs: dict[str, Any]) -> None:
        self.observations.append(int(obs["frame"]))

    def infer_actions(self) -> np.ndarray:
        return np.zeros((1, 1), dtype=np.float32)

    def finalize_episode(self, episode_info=None) -> None:
        pass


class _FakeConnection:
    def __init__(self, requests: list[dict[str, Any]]) -> None:
        self._requests = [
            _safe_packb(request, default=_pack_array) for request in requests
        ]
        self.responses: list[Any] = []

    def __iter__(self):
        return iter(self._requests)

    def send(self, value: Any) -> None:
        self.responses.append(value)


class _BlockingConnection(_FakeConnection):
    def __init__(self) -> None:
        super().__init__([])
        self.iterating = threading.Event()
        self.release = threading.Event()

    def __iter__(self):
        self.iterating.set()
        assert self.release.wait(timeout=2.0)
        return iter(())


class _HoldingConnection(_FakeConnection):
    def __init__(self, requests: list[dict[str, Any]]) -> None:
        super().__init__(requests)
        self.holding = threading.Event()
        self.release = threading.Event()

    def __iter__(self):
        yield from self._requests
        self.holding.set()
        assert self.release.wait(timeout=2.0)


class _ClosableConnection(_FakeConnection):
    def __init__(self) -> None:
        super().__init__([])
        self.closed: tuple[int, str] | None = None

    def close(self, code: int, reason: str) -> None:
        self.closed = (code, reason)


class _LifecycleModel(_RecordingModel):
    def __init__(self) -> None:
        super().__init__()
        self.initialize_count = 0
        self.finalize_count = 0
        self.close_count = 0

    def initialize_episode(self, episode_info=None) -> None:
        self.initialize_count += 1

    def finalize_episode(self, episode_info=None) -> None:
        self.finalize_count += 1

    def close(self) -> None:
        self.close_count += 1


class _SerializedInferenceModel(_RecordingModel):
    def __init__(self) -> None:
        super().__init__()
        self.current_frame: int | None = None
        self.inferred_frames: list[int] = []
        self.first_inference_started = threading.Event()
        self.release_first_inference = threading.Event()

    def ingest_observation(self, obs: dict[str, Any]) -> None:
        self.current_frame = int(obs["frame"])

    def infer_actions(self) -> np.ndarray:
        assert self.current_frame is not None
        frame = self.current_frame
        if frame == 1:
            self.first_inference_started.set()
            assert self.release_first_inference.wait(timeout=2.0)
        self.inferred_frames.append(frame)
        return np.array([[frame]], dtype=np.float32)


def _multiplexed_server(model: PolicyModel) -> PolicySpaceServer:
    return PolicySpaceServer(model, metadata={"session_mode": "multiplexed_serial"})


def test_infer_actions_does_not_duplicate_an_explicitly_ingested_observation() -> None:
    model = _RecordingModel()
    connection = _FakeConnection(
        [
            {"endpoint": "initialize_episode", "episode_id": "episode-1"},
            {"endpoint": "ingest_observation", "frame": 1},
            {"endpoint": "ingest_observation", "frame": 2},
            {"endpoint": "infer_actions", "frame": 2},
            {"endpoint": "infer_actions", "frame": 3},
            {"endpoint": "finalize_episode", "episode_id": "episode-1"},
        ]
    )

    PolicySpaceServer(model)._handle_client(connection)

    assert model.observations == [1, 2, 3]


def test_infer_actions_only_client_still_ingests_its_payload() -> None:
    model = _RecordingModel()
    connection = _FakeConnection(
        [
            {"endpoint": "initialize_episode", "episode_id": "episode-1"},
            {"endpoint": "infer_actions", "frame": 7},
        ]
    )

    PolicySpaceServer(model)._handle_client(connection)

    assert model.observations == [7]


def test_handshake_reports_v2_without_implicit_episode_initialization() -> None:
    model = _LifecycleModel()
    connection = _FakeConnection([])

    PolicySpaceServer(model)._handle_client(connection)

    metadata = _safe_unpackb(connection.responses[0])
    assert metadata["server"] == "policy_space"
    assert metadata["version"] == 2
    assert metadata["protocol_version"] == "2"
    assert isinstance(metadata["server_instance_id"], str)
    assert metadata["server_instance_id"]
    assert model.initialize_count == 0


def test_disconnect_preserves_episode_for_same_process_reconnect() -> None:
    model = _LifecycleModel()
    server = PolicySpaceServer(model)
    first = _FakeConnection(
        [
            {"endpoint": "initialize_episode", "episode_id": "episode-1"},
            {"endpoint": "ingest_observation", "frame": 1},
        ]
    )
    server._handle_client(first)

    replacement = _FakeConnection(
        [
            {"endpoint": "initialize_episode", "episode_id": "episode-1"},
            {"endpoint": "infer_actions", "frame": 1},
            {"endpoint": "finalize_episode", "episode_id": "episode-1"},
        ]
    )
    server._handle_client(replacement)

    assert model.initialize_count == 1
    assert model.finalize_count == 1
    assert model.close_count == 0


def test_new_episode_finalizes_abandoned_episode_before_initialization() -> None:
    model = _LifecycleModel()
    server = PolicySpaceServer(model)

    server._handle_client(
        _FakeConnection([{"endpoint": "initialize_episode", "episode_id": "episode-1"}])
    )
    server._handle_client(
        _FakeConnection(
            [
                {"endpoint": "initialize_episode", "episode_id": "episode-2"},
                {"endpoint": "finalize_episode", "episode_id": "episode-2"},
            ]
        )
    )

    assert model.initialize_count == 2
    assert model.finalize_count == 2


def test_retried_request_id_does_not_duplicate_history() -> None:
    model = _RecordingModel()
    server = PolicySpaceServer(model)
    server._handle_client(
        _FakeConnection(
            [
                {"endpoint": "initialize_episode", "episode_id": "episode-1"},
                {
                    "endpoint": "ingest_observation",
                    "episode_id": "episode-1",
                    "frame": 7,
                    "request_id": "request-1",
                },
            ]
        )
    )

    server._handle_client(
        _FakeConnection(
            [
                {"endpoint": "initialize_episode", "episode_id": "episode-1"},
                {
                    "endpoint": "ingest_observation",
                    "episode_id": "episode-1",
                    "frame": 7,
                    "request_id": "request-1",
                },
            ]
        )
    )

    assert model.observations == [7]


@pytest.mark.parametrize(
    "endpoint", ["reset", "update_obs", "get_action", "get_action_batch", "close"]
)
def test_v1_endpoints_are_not_accepted(endpoint: str) -> None:
    server = PolicySpaceServer(_RecordingModel())

    with pytest.raises(ValueError, match=f"Unknown endpoint: {endpoint}"):
        server._dispatch(endpoint, {})


def test_concurrent_client_is_rejected_without_touching_active_model() -> None:
    model = _LifecycleModel()
    server = PolicySpaceServer(model)
    active = _BlockingConnection()
    active_thread = threading.Thread(target=server._handle_client, args=(active,))
    active_thread.start()
    assert active.iterating.wait(timeout=1.0)

    concurrent = _ClosableConnection()
    try:
        server._handle_client(concurrent)

        assert model.initialize_count == 0
        assert model.close_count == 0
        assert concurrent.closed is not None
        assert concurrent.closed[0] == 1013
        busy_metadata = _safe_unpackb(concurrent.responses[0])
        assert busy_metadata["error"] == "server_busy"
        assert busy_metadata["retryable"] is True
    finally:
        active.release.set()
        active_thread.join(timeout=1.0)

    assert not active_thread.is_alive()
    assert model.close_count == 0

    replacement = _FakeConnection([])
    server._handle_client(replacement)
    replacement_metadata = _safe_unpackb(replacement.responses[0])
    assert "error" not in replacement_metadata
    assert model.initialize_count == 0
    assert model.close_count == 0


def test_multiplexed_clients_keep_observation_and_inference_atomic() -> None:
    model = _SerializedInferenceModel()
    server = _multiplexed_server(model)
    first = _FakeConnection(
        [
            {"endpoint": "initialize_episode", "episode_id": "episode-1"},
            {"endpoint": "infer_actions", "episode_id": "env-0", "frame": 1},
            {"endpoint": "finalize_episode", "episode_id": "episode-1"},
        ]
    )
    second = _FakeConnection(
        [
            {"endpoint": "initialize_episode", "episode_id": "episode-2"},
            {"endpoint": "infer_actions", "episode_id": "env-0", "frame": 2},
            {"endpoint": "finalize_episode", "episode_id": "episode-2"},
        ]
    )

    first_thread = threading.Thread(target=server._handle_client, args=(first,))
    second_thread = threading.Thread(target=server._handle_client, args=(second,))
    first_thread.start()
    assert model.first_inference_started.wait(timeout=1.0)
    second_thread.start()
    try:
        assert model.current_frame == 1
    finally:
        model.release_first_inference.set()
        first_thread.join(timeout=2.0)
        second_thread.join(timeout=2.0)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert model.inferred_frames == [1, 2]
    first_action = _safe_unpackb(first.responses[2], object_hook=_unpack_array)[
        "actions"
    ]
    second_action = _safe_unpackb(second.responses[2], object_hook=_unpack_array)[
        "actions"
    ]
    assert np.asarray(first_action).reshape(-1).tolist() == [1.0]
    assert np.asarray(second_action).reshape(-1).tolist() == [2.0]


def test_multiplexed_server_accepts_two_real_websocket_clients() -> None:
    model = _SerializedInferenceModel()
    policy_server = _multiplexed_server(model)
    results: dict[str, float] = {}
    errors: list[BaseException] = []
    second_initialized = threading.Event()

    def run_episode(
        episode_id: str, frame: int, initialized: threading.Event | None = None
    ) -> None:
        try:
            with connect(
                endpoint, open_timeout=2.0, max_size=None, compression=None
            ) as client:
                metadata = _safe_unpackb(client.recv())
                assert metadata["session_mode"] == "multiplexed_serial"
                client.send(
                    _safe_packb(
                        {"endpoint": "initialize_episode", "episode_id": episode_id}
                    )
                )
                assert _safe_unpackb(client.recv())["ok"] is True
                if initialized is not None:
                    initialized.set()
                client.send(
                    _safe_packb(
                        {
                            "endpoint": "infer_actions",
                            "episode_id": f"{episode_id}-env-0",
                            "frame": frame,
                        }
                    )
                )
                response = _safe_unpackb(client.recv(), object_hook=_unpack_array)
                results[episode_id] = float(
                    np.asarray(response["actions"]).reshape(-1)[0]
                )
                client.send(
                    _safe_packb(
                        {"endpoint": "finalize_episode", "episode_id": episode_id}
                    )
                )
                assert _safe_unpackb(client.recv())["ok"] is True
        except (
            BaseException
        ) as exc:  # pragma: no cover - surfaced by the final assertion
            errors.append(exc)

    with serve(
        policy_server._handle_client, "127.0.0.1", 0, max_size=None, compression=None
    ) as websocket_server:
        server_thread = threading.Thread(
            target=websocket_server.serve_forever, daemon=True
        )
        server_thread.start()
        endpoint = f"ws://127.0.0.1:{websocket_server.socket.getsockname()[1]}"
        first_thread = threading.Thread(target=run_episode, args=("episode-1", 1))
        second_thread = threading.Thread(
            target=run_episode,
            args=("episode-2", 2, second_initialized),
        )
        first_thread.start()
        assert model.first_inference_started.wait(timeout=2.0)
        second_thread.start()
        try:
            assert second_initialized.wait(timeout=2.0)
            assert model.current_frame == 1
        finally:
            model.release_first_inference.set()
            first_thread.join(timeout=2.0)
            second_thread.join(timeout=2.0)
            websocket_server.shutdown()
            server_thread.join(timeout=2.0)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert errors == []
    assert results == {"episode-1": 1.0, "episode-2": 2.0}
    assert model.inferred_frames == [1, 2]


def test_multiplexed_mode_rejects_history_ingestion() -> None:
    model = _RecordingModel()
    connection = _FakeConnection(
        [
            {"endpoint": "initialize_episode", "episode_id": "episode-1"},
            {"endpoint": "ingest_observation", "frame": 1},
        ]
    )

    _multiplexed_server(model)._handle_client(connection)

    assert model.observations == []
    assert isinstance(connection.responses[-1], str)
    assert "multiplexed_serial" in connection.responses[-1]


def test_multiplexed_reconnect_deduplicates_inference_by_episode() -> None:
    model = _RecordingModel()
    server = _multiplexed_server(model)
    request = {
        "endpoint": "infer_actions",
        "episode_id": "env-0",
        "frame": 7,
        "request_id": "request-1",
    }

    server._handle_client(
        _FakeConnection(
            [
                {"endpoint": "initialize_episode", "episode_id": "episode-1"},
                request,
            ]
        )
    )
    server._handle_client(
        _FakeConnection(
            [
                {"endpoint": "initialize_episode", "episode_id": "episode-1"},
                request,
                {"endpoint": "finalize_episode", "episode_id": "episode-1"},
            ]
        )
    )

    assert model.observations == [7]


def test_multiplexed_duplicate_live_episode_is_rejected() -> None:
    model = _LifecycleModel()
    server = _multiplexed_server(model)
    first = _HoldingConnection(
        [{"endpoint": "initialize_episode", "episode_id": "episode-1"}]
    )
    first_thread = threading.Thread(target=server._handle_client, args=(first,))
    first_thread.start()
    assert first.holding.wait(timeout=1.0)

    duplicate = _FakeConnection(
        [{"endpoint": "initialize_episode", "episode_id": "episode-1"}]
    )
    try:
        server._handle_client(duplicate)
        assert isinstance(duplicate.responses[-1], str)
        assert "already owned" in duplicate.responses[-1]
        assert model.initialize_count == 0
        assert model.finalize_count == 0
    finally:
        first.release.set()
        first_thread.join(timeout=1.0)

    assert not first_thread.is_alive()


def test_multiplexed_disconnected_session_cache_is_bounded() -> None:
    server = _multiplexed_server(_RecordingModel())

    for index in range(65):
        server._handle_client(
            _FakeConnection(
                [{"endpoint": "initialize_episode", "episode_id": f"episode-{index}"}]
            )
        )

    assert len(server._multiplexed_sessions) == 64
    assert "episode-0" not in server._multiplexed_sessions
    assert "episode-64" in server._multiplexed_sessions

    evicted_resume = _FakeConnection(
        [
            {
                "endpoint": "initialize_episode",
                "episode_id": "episode-0",
                "resume_only": True,
            }
        ]
    )
    server._handle_client(evicted_resume)
    assert isinstance(evicted_resume.responses[-1], str)
    assert "resume state is unavailable" in evicted_resume.responses[-1]
    assert "episode-0" not in server._multiplexed_sessions
